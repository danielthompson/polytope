//
// Created by daniel on 5/15/20.
//

#include "path_tracer.cuh"
#include "common_device_functions.cuh"

namespace Polytope {
   
   struct device_pointers {
      struct DeviceCamera* device_camera;
      
      struct DeviceMesh* device_meshes;
      unsigned int num_meshes;

      struct DeviceSamples* device_samples;
   };
   
   struct device_intersection {
      float t;
      int mesh_index;
      int face_index;
      float3 normal;
      float3 hit_point;
      bool hits;
   };
   
   __device__ device_intersection linear_intersect(struct device_pointers device_pointers, const float3 &ro, const float3 rd) {
      
      unsigned int hit_mesh_index = 0;
      
      device_intersection intersection {
            3.40282e+038f,
            0,
            0,
            {0.f, 0.f, 0.f},
            {0.f, 0.f, 0.f},
            false
      };
      
      // for each mesh on the device
      for (unsigned int mesh_index = 0; mesh_index < device_pointers.num_meshes; mesh_index++) {
         DeviceMesh mesh = device_pointers.device_meshes[mesh_index];
         
         const unsigned int v1_index_offset = mesh.num_faces;
         const unsigned int v2_index_offset = mesh.num_faces * 2;

         //unsigned int hit_face_index = 0;
         
         // for each face in the mesh
         for (int face_index = 0; face_index < mesh.num_faces; face_index++) {
            const unsigned int v1index = face_index + v1_index_offset;
            const unsigned int v2index = face_index + v2_index_offset;

            const float3 v0 = { mesh.x[face_index], mesh.y[face_index], mesh.z[face_index] };
            const float3 v1 = { mesh.x[v1index], mesh.y[v1index], mesh.z[v1index] };
            const float3 v2 = { mesh.x[v2index], mesh.y[v2index], mesh.z[v2index] };

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

            if (ft <= 0 || ft > intersection.t) {
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
            intersection.hits = true;
            intersection.t = ft;
            intersection.face_index = face_index;
            intersection.mesh_index = mesh_index;
         }
      }
      
      if (!intersection.hits)
         return intersection;
      
      // TODO this is (very) non-optimal
      intersection.hit_point = ro + rd * intersection.t;

      // calculate normal at hit point
      DeviceMesh mesh_hit = device_pointers.device_meshes[intersection.mesh_index];

      const unsigned int v1_index = intersection.face_index + mesh_hit.num_faces;
      const unsigned int v2_index = intersection.face_index + mesh_hit.num_faces * 2;

      const float3 v0 = {device_pointers.device_meshes[intersection.mesh_index].x[intersection.face_index],
                         device_pointers.device_meshes[intersection.mesh_index].y[intersection.face_index],
                         device_pointers.device_meshes[intersection.mesh_index].z[intersection.face_index]};
      const float3 v1 = {device_pointers.device_meshes[intersection.mesh_index].x[v1_index],
                         device_pointers.device_meshes[intersection.mesh_index].y[v1_index],
                         device_pointers.device_meshes[intersection.mesh_index].z[v1_index]};
      const float3 v2 = {device_pointers.device_meshes[intersection.mesh_index].x[v2_index],
                         device_pointers.device_meshes[intersection.mesh_index].y[v2_index],
                         device_pointers.device_meshes[intersection.mesh_index].z[v2_index]};

      const float3 e0 = v1 - v0;
      const float3 e1 = v2 - v1;
      const float3 e2 = v0 - v2;
      float3 n = cross(e0, e1);

      // flip normal if needed
      const float ray_dot_normal = dot(rd, n);
      const float flip_factor = ray_dot_normal > 0 ? -1 : 1;
      n *= flip_factor;

      normalize(n);
      intersection.normal = n;
   }
   
   __global__ void path_trace_kernel(const unsigned int width, const unsigned int height, struct device_pointers device_pointers) {
      // loop over pixels
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      const unsigned int pixel_x = pixel_index % width;
      const unsigned int pixel_y = pixel_index / width;

      float3 ro = { device_pointers.device_camera->ox[pixel_index], device_pointers.device_camera->oy[pixel_index], device_pointers.device_camera->oz[pixel_index] };
      float3 rd = { device_pointers.device_camera->dx[pixel_index], device_pointers.device_camera->dy[pixel_index], device_pointers.device_camera->dz[pixel_index] };

      unsigned int num_bounces = 0;
      
      float3 src = { 1.f, 1.f, 1.f};
      
      while (true) {
         device_intersection intersection = linear_intersect(device_pointers, ro, rd);

         if (!intersection.hits) {
            device_pointers.device_samples->r[pixel_index] = src.x * 255.f;
            device_pointers.device_samples->g[pixel_index] = src.y * 255.f;
            device_pointers.device_samples->b[pixel_index] = src.z * 255.f;
            return;
         } 
      
         // todo hit light
      
         

         // todo reflect / bounce according to BRDF


         device_pointers.device_samples->r[pixel_index] = 128.f;
         device_pointers.device_samples->g[pixel_index] = 128.f;
         device_pointers.device_samples->b[pixel_index] = 128.f;
         return;
      
      }
      
   }
   
   void PathTracerKernel::Trace() {

      struct device_pointers device_pointers {
         memory_manager->device_camera,
         memory_manager->meshes,
         memory_manager->num_meshes,
         memory_manager->device_samples
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
