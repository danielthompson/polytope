//
// Created by daniel on 5/14/20.
//

#include <cstdio>
#include <utility>
#include "generate_rays.cuh"

namespace Polytope {
   
   __constant__ float camera_to_world_matrix[16];

   RayGeneratorKernel::RayGeneratorKernel(Polytope::AbstractScene *scene, std::shared_ptr<Polytope::GPUMemoryManager> memory_manager) 
      : scene(scene), memory_manager(std::move(memory_manager)) {
      
      cudaError_t error = cudaMemcpyToSymbol(camera_to_world_matrix, scene->Camera->CameraToWorld.Matrix.Matrix, 16 * sizeof(float), 0, cudaMemcpyHostToDevice);
      if (error != cudaSuccess) {
         fprintf(stderr, "Failed to copy camera matrix to device (error code [%s])!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
   }
   
   RayGeneratorKernel::~RayGeneratorKernel() {
      // TODO
   }
   
   __device__ void normalize(float3 &v) {
      const float one_over_length = 1.f / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
      v.x *= one_over_length;
      v.y *= one_over_length;
      v.z *= one_over_length;
   }

   __device__ float3 normalize(const float3 &v) {
      const float one_over_length = 1.f / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
      return {
         v.x * one_over_length,
         v.y * one_over_length,
         v.z * one_over_length
      };
   }
   
   __device__ float3 matrix_apply_point(const float* d_matrix, const float3 d_p) {
      const float w = d_p.x * d_matrix[12] + d_p.y * d_matrix[13] + d_p.z * d_matrix[14] + d_matrix[15];
      const float divisor = 1.f / w;
      
      return {
            (d_p.x * d_matrix[0] + d_p.y * d_matrix[1] + d_p.z * d_matrix[2] + d_matrix[3]) * divisor,
            (d_p.x * d_matrix[4] + d_p.y * d_matrix[5] + d_p.z * d_matrix[6] + d_matrix[7]) * divisor,
            (d_p.x * d_matrix[8] + d_p.y * d_matrix[9] + d_p.z * d_matrix[10] + d_matrix[11]) * divisor
      };
   }

   __device__ float3 matrix_apply_vector(const float* d_matrix, const float3 d_v) {
      return {
            d_v.x * d_matrix[0] + d_v.y * d_matrix[1] + d_v.z * d_matrix[2],
            d_v.x * d_matrix[4] + d_v.y * d_matrix[5] + d_v.z * d_matrix[6],
            d_v.x * d_matrix[8] + d_v.y * d_matrix[9] + d_v.z * d_matrix[10]
      };
   }
   
   __global__ void
   generate_rays_kernel(RayGeneratorKernel::params p) {
      constexpr float PIOver360 = M_PI / 360.f;
      
      const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
      const unsigned int pixel_x = index % p.width;
      const unsigned int pixel_y = index / p.width;
      
      const float pixel_ndc_x = (float)pixel_x / (float)p.width;
      const float pixel_ndc_y = (float)pixel_y / (float)p.height;
      
      const float aspect = (float)p.width / (float)p.height;
      const float tan_fov_half = tan(p.fov * PIOver360);

      const float3 camera_origin = {0, 0, 0};
      float3 camera_direction = {
            (2 * pixel_ndc_x - 1) * aspect * tan_fov_half,
            (1 - 2 * pixel_ndc_y) * tan_fov_half,
            // TODO -1 for right-handed
            1
      };
      
      normalize(camera_direction);
      
      const float3 world_origin = matrix_apply_point(camera_to_world_matrix, camera_origin);
      float3 world_direction = matrix_apply_vector(camera_to_world_matrix, camera_direction);
      
      normalize(world_direction);
      
      p.d_ox[index] = world_origin.x;
      p.d_oy[index] = world_origin.y;
      p.d_oz[index] = world_origin.z;

      p.d_dx[index] = world_direction.x;
      p.d_dy[index] = world_direction.y;
      p.d_dz[index] = world_direction.z;
   }
   
   void RayGeneratorKernel::GenerateRays() {
      
      cudaError_t error = cudaSuccess;

      error = cudaSetDevice(0);
      if (error != cudaSuccess) {
         fprintf(stderr, "Failed to get set device to 0 (error code [%s])!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }

      std::shared_ptr<CameraRays> rays = memory_manager->MallocCameraRays();

      struct params kernel_params = {
            memory_manager->width,
            memory_manager->height,
            scene->Camera->Settings.FieldOfView,
            rays->d_o[0], rays->d_o[1], rays->d_o[2], rays->d_d[0], rays->d_d[1], rays->d_d[2]
      };
      
      constexpr unsigned int threadsPerBlock = 256;
      const unsigned int blocksPerGrid = (memory_manager->num_pixels + threadsPerBlock - 1) / threadsPerBlock;

      error = cudaSuccess;
      generate_rays_kernel<<<blocksPerGrid, threadsPerBlock>>>(kernel_params);

      cudaDeviceSynchronize();
      error = cudaGetLastError();   

      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to launch generate_ray_kernel (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
   }

   void RayGeneratorKernel::CheckRays() {
      const size_t num_bytes = sizeof(float) * memory_manager->num_pixels;
      float* h_ox = (float *)calloc(memory_manager->num_pixels, sizeof(float));
      
      cudaError_t error = cudaMemcpy(h_ox, memory_manager->camera_rays->d_o[0], num_bytes, cudaMemcpyDeviceToHost);
      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to check rays (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      free(h_ox);
   }
}
