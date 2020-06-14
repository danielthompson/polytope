////
//// Created by daniel on 5/14/20.
////
//
//#include <cstdio>
//#include <cassert>
//#include <utility>
//#include "generate_rays.cuh"
//#include "common_device_functions.cuh"
//
//
//namespace poly {
//   
//   
//
//   RayGeneratorKernel::RayGeneratorKernel(poly::Scene *scene, std::shared_ptr<poly::GPUMemoryManager> memory_manager) 
//      : scene(scene), memory_manager(std::move(memory_manager)) {
//      
////      cudaError_t error = cudaMemcpyToSymbol(camera_to_world_matrix, scene->Camera->CameraToWorld.Matrix.Matrix, 16 * sizeof(float), 0, cudaMemcpyHostToDevice);
////      if (error != cudaSuccess) {
////         fprintf(stderr, "Failed to copy camera matrix to device (error code [%s])!\n", cudaGetErrorString(error));
////         exit(EXIT_FAILURE);
////      }
//   }
//   
//   RayGeneratorKernel::~RayGeneratorKernel() {
//      // TODO
//   }
//   
//   __global__ void
//   generate_rays_kernel(RayGeneratorKernel::params p) {
//      constexpr float PIOver360 = M_PI / 360.f;
//
//      const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
//      const unsigned int pixel_x = index % p.width;
//      const unsigned int pixel_y = index / p.width;
//
//      const float pixel_ndc_x = (float) pixel_x / (float) p.width;
//      const float pixel_ndc_y = (float) pixel_y / (float) p.height;
//
//      const float aspect = (float) p.width / (float) p.height;
//      const float tan_fov_half = tan(p.fov * PIOver360);
//
//      const float3 camera_origin = {0, 0, 0};
//      float3 camera_direction = {
//            (2 * pixel_ndc_x - 1) * aspect * tan_fov_half,
//            (1 - 2 * pixel_ndc_y) * tan_fov_half,
//            // TODO -1 for right-handed
//            1
//      };
//
//      normalize(camera_direction);
//
////      const float3 world_origin = matrix_apply_point(camera_to_world_matrix, camera_origin);
////      float3 world_direction = matrix_apply_vector(camera_to_world_matrix, camera_direction);
//
//      normalize(world_direction);
//
//      assert(p.camera != nullptr);
//      assert(p.camera->ox != nullptr);
//      assert(p.camera->oy != nullptr);
//      assert(p.camera->oz != nullptr);
//      assert(p.camera->dx != nullptr);
//      assert(p.camera->dy != nullptr);
//      assert(p.camera->dz != nullptr);
//      
//      p.camera->ox[index] = world_origin.x;
//      p.camera->oy[index] = world_origin.y;
//      p.camera->oz[index] = world_origin.z;
//      p.camera->dx[index] = world_direction.x;
//      p.camera->dy[index] = world_direction.y;
//      p.camera->dz[index] = world_direction.z;
//   }
//   
//   void RayGeneratorKernel::GenerateRays() {
//      
//      cudaError_t error = cudaSuccess;
//
//      struct params kernel_params = {
//            memory_manager->width,
//            memory_manager->height,
//            scene->Camera->Settings.FieldOfView,
//            memory_manager->device_camera
//      };
//      
//      constexpr unsigned int threadsPerBlock = 256;
//      const unsigned int blocksPerGrid = (memory_manager->num_pixels + threadsPerBlock - 1) / threadsPerBlock;
//
//      error = cudaSuccess;
//      cudaDeviceSynchronize();
//      error = cudaGetLastError();
//
//      if (error != cudaSuccess)
//      {
//         fprintf(stderr, "Error before launch generate_ray_kernel (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//      }
//      
//      generate_rays_kernel<<<blocksPerGrid, threadsPerBlock>>>(kernel_params);
//
//      cudaDeviceSynchronize();
//      error = cudaGetLastError();   
//
//      if (error != cudaSuccess)
//      {
//         fprintf(stderr, "Failed to launch generate_ray_kernel (error code %s)!\n", cudaGetErrorString(error));
//         exit(EXIT_FAILURE);
//      }
//   }
//
//   void RayGeneratorKernel::CheckRays() {
////      const size_t num_bytes = sizeof(float) * memory_manager->num_pixels;
////      float* h_ox = (float *)calloc(memory_manager->num_pixels, sizeof(float));
////      
////      cudaError_t error = cudaMemcpy(h_ox, memory_manager->device_camera->ox, num_bytes, cudaMemcpyDeviceToHost);
////      if (error != cudaSuccess)
////      {
////         fprintf(stderr, "Failed to check rays (error code %s)!\n", cudaGetErrorString(error));
////         exit(EXIT_FAILURE);
////      }
////      
////      free(h_ox);
//   }
//}
