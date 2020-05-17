//
// Created by daniel on 5/15/20.
//

#include "path_tracer.cuh"

namespace Polytope {
   
   struct device_pointers {
      float* ox;
      float* oy;
      float* oz;
      
      float* dx;
      float* dy;
      float* dz;
      
      float* mx;
      float* my;
      float* mz;
      
      size_t num_faces;
      
      float* samples;
   };

   __device__ float3 operator-(const float3 &a, const float3 &b) {
      return {a.x - b.x, a.y - b.y, a.z - b.z};
   }

   __device__ float3 operator+(const float3 &a, const float3 &b) {
      return {a.x + b.x, a.y + b.y, a.z + b.z};
   }

   __device__ float3 operator*(const float3 &a, const float t) {
      return {a.x * t, a.y * t, a.z * t};
   }
   
   __device__ void operator*=(float3 &a, const float t) {
      a.x *= t;
      a.y *= t;
      a.z *= t;
   }

   __device__ float3 cross(const float3 &a, const float3 &b) {
      return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
      };
   }
   
   __device__ float dot(const float3 &a, const float3 &b) {
      return a.x * b.x + a.y * b.y + a.z * b.z;
   }
   
   __global__ void path_trace_kernel(const unsigned int width, const unsigned int height, struct device_pointers device_pointers) {
      // loop over pixels
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      const unsigned int pixel_x = pixel_index % width;
      const unsigned int pixel_y = pixel_index / width;

      const float3 ro = { device_pointers.ox[pixel_index], device_pointers.oy[pixel_index], device_pointers.oz[pixel_index] };
      const float3 rd = { device_pointers.dx[pixel_index], device_pointers.dy[pixel_index], device_pointers.dz[pixel_index] };

      const unsigned int v1_index_offset = device_pointers.num_faces;
      const unsigned int v2_index_offset = device_pointers.num_faces * 2;

      device_pointers.samples[pixel_index] = 255.f;

      for (int face_index = 0; face_index < device_pointers.num_faces; face_index++) {
         const unsigned int v1index = face_index + v1_index_offset;
         const unsigned int v2index = face_index + v2_index_offset;

         const float3 v0 = { device_pointers.mx[face_index], device_pointers.my[face_index], device_pointers.mz[face_index] };
         const float3 v1 = { device_pointers.mx[v1index], device_pointers.my[v1index], device_pointers.mz[v1index] };
         const float3 v2 = { device_pointers.mx[v2index], device_pointers.my[v2index], device_pointers.mz[v2index] };

         const float3 e0 = v1 - v0;
         const float3 e1 = v2 - v1;
         float3 pn = cross(e0, e1);

         // TODO use cuda intrinsic         
         const float oneOverLength = 1.0f / sqrt(dot(pn, pn));
         pn *= oneOverLength;

         const float divisor = dot(pn, rd);
         if (divisor == 0.0f) {
            // parallel
            continue;
         }

         const float ft = (dot(pn, v0 - ro)) / divisor;

         if (ft <= 0) {
            continue;
         }

         const float3 hp = ro + rd * ft;
         const float3 e2 = v0 - v2;

         const float3 p0 = hp - v0;
         const float3 cross0 = cross(e0, p0);
         const float normal0 = dot(cross0, pn);
         const bool pos0 = normal0 > 0;

         if (!pos0) {
            continue;
         }

         const float3 p1 = hp - v1;
         const float3 cross1 = cross(e1, p1);
         const float normal1 = dot(cross1, pn);
         const bool pos1 = normal1 > 0;

         if (!pos1) {
            continue;
         }

         const float3 p2 = hp - v2;
         const float3 cross2 = cross(e2, p2);
         const float normal2 = dot(cross2, pn);
         const bool pos2 = normal2 > 0;

         if (!pos2) {
            continue;
         }

         // temp
         device_pointers.samples[pixel_index] = 128.f;
         return;
      }
   }
   
   void PathTracerKernel::Trace() {

      struct device_pointers device_pointers {
         memory_manager->camera_rays->d_o[0],
         memory_manager->camera_rays->d_o[1],
         memory_manager->camera_rays->d_o[2],
         memory_manager->camera_rays->d_d[0],
         memory_manager->camera_rays->d_d[1],
         memory_manager->camera_rays->d_d[2],
         memory_manager->meshes_on_device[0]->d_x,
         memory_manager->meshes_on_device[0]->d_y,
         memory_manager->meshes_on_device[0]->d_z,
         memory_manager->meshes_on_device[0]->bytes / 4L,
         memory_manager->d_samples
      };
      
      constexpr unsigned int threadsPerBlock = 256;
      const unsigned int blocksPerGrid = (memory_manager->num_pixels + threadsPerBlock - 1) / threadsPerBlock;

      const unsigned int width = memory_manager->width;
      const unsigned int height = memory_manager->height;
      
      cudaError_t error = cudaSuccess;
      path_trace_kernel<<<blocksPerGrid, threadsPerBlock>>>(width, height, device_pointers);

      cudaDeviceSynchronize();
      error = cudaGetLastError();

      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to launch generate_ray_kernel (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      //Intersection intersection = Scene->GetNearestShape(current_ray, x, y);
   }
}
