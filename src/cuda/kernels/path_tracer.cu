
// Created by daniel on 5/15/20.
//


#include <curand_kernel.h>
#include <cassert>
#include "path_tracer.cuh"
#include "common_device_functions.cuh"
#include "../check_error.h"


namespace Polytope {

   __constant__ float camera_to_world_matrix[16];
   
   struct device_pointers {
      struct DeviceCamera* device_camera;
      
      struct DeviceMesh* device_meshes;
      unsigned int num_meshes;

      struct Samples* device_samples;
   };
   
   struct device_intersection {
      float t;
      int mesh_index;
      int face_index;
      float3 normal;
      float3 hit_point;
      float3 tangent1;
      float3 tangent2;
      bool hits;
   };
   
   __device__ bool aabb_hits(const DeviceMesh &mesh, const float3 ro, const float3 rd ) {
      float maxBoundFarT = Polytope::FloatMax;
      float minBoundNearT = 0;

      // TODO
      //const float gammaMultiplier = 1 + 2 * Polytope::Gamma(3);

      // X
      float divisor = 1.f / rd.x;
      
      float tNear = (mesh.aabb[0] - ro.x) * divisor;
      float tFar = (mesh.aabb[3] - ro.x) * divisor;

      float swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      // TODO
      //tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;
      if (minBoundNearT > maxBoundFarT) {
         return false;
      }

      // Y
      divisor = 1.f / rd.y;
      tNear = (mesh.aabb[1] - ro.y) * divisor;
      tFar = (mesh.aabb[4] - ro.y) * divisor;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      // TODO
      // tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      if (minBoundNearT > maxBoundFarT) {
         return false;
      }

      // z
      divisor = 1.f / rd.z;
      tNear = (mesh.aabb[2] - ro.z) * divisor;
      tFar = (mesh.aabb[5] - ro.z) * divisor;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      // TODO
      // tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      return minBoundNearT <= maxBoundFarT;
   }
   
   __device__ void linear_intersect(
         struct device_intersection* sample_intersections,
         const float3* sample_origins,
         const float3* sample_directions,
         const unsigned int num_samples,
         struct device_pointers device_pointers, 
         // TODO use something more efficient than bool array
         bool* active_mask,
         bool debug) {
      
      // const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

      __shared__ float v0x[32];
      __shared__ float v0y[32];
      __shared__ float v0z[32];

      __shared__ float v1x[32];
      __shared__ float v1y[32];
      __shared__ float v1z[32];

      __shared__ float v2x[32];
      __shared__ float v2y[32];
      __shared__ float v2z[32];
      
      // for each mesh on the device
      for (unsigned int mesh_index = 0; mesh_index < device_pointers.num_meshes; mesh_index++) {
         DeviceMesh mesh = device_pointers.device_meshes[mesh_index];

         bool all_samples_miss = true;
         
         // check to see if all samples miss the mesh's AABB
         for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {
            bool hits_aabb = aabb_hits(mesh, sample_origins[sample_index], sample_directions[sample_index]);
            if (hits_aabb)
               all_samples_miss = false;
         }

         // if all samples miss, we can safely skip the mesh
         if (__all_sync(0xFFFFFFFF, all_samples_miss))
            continue;
         
         const float* mesh_x = mesh.x;
         const float* mesh_y = mesh.y;
         const float* mesh_z = mesh.z;
         
         // for each face in the mesh
         for (unsigned int face_index = 0; face_index < mesh.num_faces; face_index++) {

            // every 32 faces, have the threads in the warp coordinate to load the next 32 faces' vertices from global to shared memory
            const unsigned int shared_index = face_index & 0x0000001Fu;
            const unsigned int thread_face_index = face_index + threadIdx.x;
            if (shared_index == 0 && thread_face_index < mesh.num_faces) {
               v0x[threadIdx.x] = mesh_x[thread_face_index];
               v0y[threadIdx.x] = mesh_y[thread_face_index];
               v0z[threadIdx.x] = mesh_z[thread_face_index];
               
               const unsigned int v1_index = thread_face_index + mesh.num_faces;
               v1x[threadIdx.x] = mesh_x[v1_index];
               v1y[threadIdx.x] = mesh_y[v1_index];
               v1z[threadIdx.x] = mesh_z[v1_index];

               const unsigned int v2_index = thread_face_index + (mesh.num_faces * 2);
               v2x[threadIdx.x] = mesh_x[v2_index];
               v2y[threadIdx.x] = mesh_y[v2_index];
               v2z[threadIdx.x] = mesh_z[v2_index];
            }
            
            __syncthreads();
            
            // skip this face if all the samples are done with it
            bool all_samples_done = true;
            for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {
               if (!active_mask[sample_index])
                  all_samples_done = false;
            }
            if (!all_samples_done) {
               continue;
            }

            //            assert(v0x[shared_index] == mesh.x[face_index]);
//            assert(v0y[face_index % 32] == mesh.y[face_index]);
//            assert(v0z[face_index % 32] == mesh.z[face_index]);
//
//            assert(v1x[face_index % 32] == mesh.x[face_index + v1_index_offset]);
//            assert(v1y[face_index % 32] == mesh.y[face_index + v1_index_offset]);
//            assert(v1z[face_index % 32] == mesh.z[face_index + v1_index_offset]);
//
//            assert(v2x[face_index % 32] == mesh.x[face_index + v2_index_offset]);
//            assert(v2y[face_index % 32] == mesh.y[face_index + v2_index_offset]);
//            assert(v2z[face_index % 32] == mesh.z[face_index + v2_index_offset]);

            const float3 v0 = make_float3(v0x[shared_index], v0y[shared_index], v0z[shared_index]);
            //const unsigned int v1index = face_index + v1_index_offset;
            const float3 v1 = {v1x[shared_index], v1y[shared_index], v1z[shared_index]};
            const float3 e0 = v1 - v0;
            //const unsigned int v2index = face_index + v2_index_offset;
            const float3 v2 = {v2x[shared_index], v2y[shared_index], v2z[shared_index]};
            const float3 e1 = v2 - v1;
            float3 pn = cross(e0, e1);

            //const float oneOverLength = 1.0f / sqrt(dot(pn, pn));
            pn *= __frsqrt_rn(dot(pn, pn));
//            pn *= (1.0f / sqrtf(dot(pn, pn)));

            
            // TODO do all samples for a vertex batch at once, since loading the vertices from global to shared memory is the bottleneck
            for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {
               // don't intersect a sample if it's not active
               if (!active_mask[sample_index])
                  continue;

               float ft = INFINITY;
               float3 ro = sample_origins[sample_index];
               float3 rd = sample_directions[sample_index];
//               if (debug) {
//                  printf("intersecting:\n");
//                  printf("o: %f %f %f\n", ro.x, ro.y, ro.z);
//                  printf("d: %f %f %f\n", rd.x, rd.y, rd.z);
//               }
               
               device_intersection& intersection = sample_intersections[sample_index];
               {
                  const float divisor = dot(pn, rd);
                  if (divisor == 0.0f) {
                     // parallel
                     continue;
                  }

                  ft = (dot(pn, v0 - ro)) / divisor;
               }

               if (ft <= 0 || ft > intersection.t) {
                  continue;
               }

//            if (face_index == 1068 && debug) {
//               printf("t: %f\n", ft);
//            }

               const float3 hp = ro + rd * ft;

               {
                  float3 p = hp - v0;
                  float3 cross_ep = cross(e0, p);
                  float normal = dot(cross_ep, pn);
                  if (normal <= 0)
                     continue;

                  p = hp - v1;
                  cross_ep = cross(e1, p);
                  normal = dot(cross_ep, pn);
                  if (normal <= 0)
                     continue;

                  const float3 e2 = v0 - v2;
                  p = hp - v2;
                  cross_ep = cross(e2, p);
                  normal = dot(cross_ep, pn);
                  if (normal <= 0)
                     continue;
               }

               // temp
               intersection.hits = true;
               intersection.t = ft;
               intersection.face_index = face_index;
               intersection.mesh_index = mesh_index;
            }
         }
      }

      for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {
         device_intersection& intersection = sample_intersections[sample_index];
         if (intersection.hits) {
            float3 ro = sample_origins[sample_index];
            float3 rd = sample_directions[sample_index];

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

            // TODO this is (very) non-optimal
            intersection.hit_point = make_float3(
                  fma(rd.x, intersection.t, ro.x),
                  fma(rd.y, intersection.t, ro.y),
                  fma(rd.z, intersection.t, ro.z)
            );

//            if (debug) {
//               printf("hit point:\n");
//               printf("o: %f %f %f\n", intersection.hit_point.x, intersection.hit_point.y, intersection.hit_point.z);
//            }
            
            // offset origin - hack - temp
            intersection.hit_point.x = fma(n.x, 0.002f, intersection.hit_point.x);
            intersection.hit_point.y = fma(n.y, 0.002f, intersection.hit_point.y);
            intersection.hit_point.z = fma(n.z, 0.002f, intersection.hit_point.z);

            const float edge0dot = abs(dot(e0, e1));
            const float edge1dot = abs(dot(e1, e2));
            const float edge2dot = abs(dot(e2, e0));

            if (edge0dot > edge1dot && edge0dot > edge2dot) {
               intersection.tangent1 = e0;
            } else if (edge1dot > edge0dot && edge1dot > edge2dot) {
               intersection.tangent1 = e1;
            } else {
               intersection.tangent1 = e2;
            }

            intersection.tangent2 = cross(intersection.tangent1, n);

            normalize(intersection.tangent1);
            normalize(intersection.tangent2);
         }
      }
   }
   
   __global__ void path_trace_kernel(const unsigned int width, const unsigned int height, struct device_pointers device_pointers) {
      // loop over pixels
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      const unsigned int pixel_x = pixel_index % width;
      const unsigned int pixel_y = pixel_index / width;

      // generate camera rays

      constexpr float PIOver360 = M_PI / 360.f;
      constexpr unsigned int samples = 1 * 1;
      
      const float aspect = (float) width / (float) height;
      const float tan_fov_half = tan(device_pointers.device_camera->fov * PIOver360);

      const float3 camera_origin = {0, 0, 0};
      const float3 world_origin_orig = matrix_apply_point(camera_to_world_matrix, camera_origin);
      
      float3 sample_sum = make_float3(0, 0, 0);
      
      // samples
      const int root_samples = ceil(sqrtf(samples));
      const float offset_step = 1.f / (float)root_samples;
      const float offset_step_initial = offset_step / 2.f;

      bool debug = false;
      if (pixel_x == 196 && pixel_y == 179) {
         debug = true;
      }

      curandState state;
      curand_init(pixel_index, pixel_index, 0, &state);
      
      float3 sample_origins[samples];
      float3 sample_directions[samples];
      device_intersection sample_intersections[samples];
      
      for (unsigned int x = 0; x < root_samples; x++) {
         const float offset_x = offset_step_initial + ((float) x * offset_step);
         const float pixel_ndc_x = (float) ((float) pixel_x + offset_x) / (float) width;

         for (unsigned int y = 0; y < root_samples; y++) {
            const float pixel_ndc_y =
                  (float) ((float) pixel_y + offset_step_initial + ((float) y * offset_step)) / (float) height;

            const unsigned int sample_index = x * root_samples + y;
            
            float3 world_origin = world_origin_orig;
            float3 camera_direction = {
                  (2 * pixel_ndc_x - 1) * aspect * tan_fov_half,
                  (1 - 2 * pixel_ndc_y) * tan_fov_half,
                  // TODO -1 for right-handed
                  1
            };
            normalize(camera_direction);
            float3 world_direction = matrix_apply_vector(camera_to_world_matrix, camera_direction);
            normalize(world_direction);
            sample_origins[sample_index] = world_origin;
            sample_directions[sample_index] = world_direction;
            
//            sample_intersections[sample_index].face_index = 0;
//            sample_intersections[sample_index].mesh_index = 0;
//            sample_intersections[sample_index].hit_point = { 0.f, 0.f, 0.f };
//            sample_intersections[sample_index].t = INFINITY;
//            sample_intersections[sample_index].hits = false;
         }
      }
            
      unsigned int num_bounces = 0;
      float3 src[samples] = { { 1.0f, 1.0f, 1.0f} };

      bool active[samples] = { true };
      
      while (true) {
//         if (debug) {
//            printf("thread %i: bounce %i\n", threadIdx.x, num_bounces);
//         }
         
         if (num_bounces > 5) {
            // kill any samples that are still active
            for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
               if (active[sample_index]) 
                  src[sample_index] = { 0, 0, 0};
            }
            break;
         }
         
         // if no samples are active in this warp, we're done
         bool any_sample_active = false;
         for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
            if (active[sample_index])
               any_sample_active = true;
         }
         
         if (__all_sync(0xFFFFFFFF, !any_sample_active)) {
//            if (debug) {
//               printf("thread %i: exited loop before bounce %i (all non-active)\n", threadIdx.x, num_bounces);
//            }
            break;
         }

         // reset all intersections
         for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
            sample_intersections[sample_index].face_index = 0;
            sample_intersections[sample_index].mesh_index = 0;
            sample_intersections[sample_index].hit_point = { 0.f, 0.f, 0.f };
            sample_intersections[sample_index].t = INFINITY;
            sample_intersections[sample_index].hits = false;
         }
         
         linear_intersect(
            sample_intersections,
            sample_origins,
            sample_directions,
            samples,
            device_pointers, 
            active,
            debug);
         
         // process each sample
         for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
            if (active[sample_index]) {
//               if (debug) {
//                  printf("bounce %i: \n", num_bounces);
//                  printf("t: %f\n", intersection.t);
//               }
               device_intersection intersection = sample_intersections[sample_index];
               float3 ro = sample_origins[sample_index];
               float3 rd = sample_directions[sample_index];
               
               if (!intersection.hits) {
//                  if (debug) {
//                     printf("no hit at bounce: %i\n", num_bounces);
//                     printf("o: %f %f %f\n", ro.x, ro.y, ro.z);
//                     printf("d: %f %f %f\n", rd.x, rd.y, rd.z);
//                  }
                  active[sample_index] = false;
                  //printf("thread %i: became inactive at bounce %i (no hit)\n", threadIdx.x, num_bounces);
                  continue;
               }

//               if (debug) {
//                  printf("hit face index %i\n", intersection.face_index);
//                  printf("o: %f %f %f\n", ro.x, ro.y, ro.z);
//                  printf("d: %f %f %f\n", rd.x, rd.y, rd.z);
//               }

               // todo hit light

               // TODO put this into a lambert_brdf function
               // const Polytope::Vector local_incoming = intersection.WorldToLocal(current_ray.Direction);
               float3 local_incoming = normalize(make_float3(dot(rd, intersection.tangent1), dot(rd, intersection.normal), dot(rd, intersection.tangent2)));

               // mirror brdf
//               const float3 local_outgoing = make_float3(local_incoming.x, -local_incoming.y, local_incoming.z); 
//               if (debug) {
//                  printf("li: %f %f %f\n", local_incoming.x, local_incoming.y, local_incoming.z);
//                  printf("lo: %f %f %f\n", local_outgoing.x, local_outgoing.y, local_outgoing.z);
//
//               }

               // lambert brdf

               float u0 = curand_uniform(&state);
               float u1 = curand_uniform(&state);
               float3 local_outgoing = cosine_sample_hemisphere(u0, u1);

               // const Polytope::Vector world_outgoing = intersection.LocalToWorld(local_outgoing);
               float3 world_outgoing = normalize(make_float3(
                     intersection.tangent1.x * local_outgoing.x + intersection.normal.x * local_outgoing.y + intersection.tangent2.x * local_outgoing.z,
                     intersection.tangent1.y * local_outgoing.x + intersection.normal.y * local_outgoing.y + intersection.tangent2.y * local_outgoing.z,
                     intersection.tangent1.z * local_outgoing.x + intersection.normal.z * local_outgoing.y + intersection.tangent2.z * local_outgoing.z
               ));
//               if (debug) {
//                  printf("n: %f %f %f\n", intersection.normal.x, intersection.normal.y, intersection.normal.z);
//                  printf("o: %f %f %f\n", intersection.hit_point.x, intersection.hit_point.y, intersection.hit_point.z);
//                  printf("d: %f %f %f\n", world_outgoing.x, world_outgoing.y, world_outgoing.z);
//
//               }

               // TOOD offset
               sample_origins[sample_index] = intersection.hit_point;
               sample_directions[sample_index] = world_outgoing;
               // TODO replace with BRDF refl
               src[sample_index] = src[sample_index] * make_float3(0.75f, 0.5f, 0.5f);

//               if (debug) {
//                  printf("bounce ray:\n");
//                  printf("o: %f %f %f\n", sample_origins[sample_index].x, sample_origins[sample_index].y, sample_origins[sample_index].z);
//                  printf("d: %f %f %f\n", sample_directions[sample_index].x, sample_directions[sample_index].y, sample_directions[sample_index].z);
//               }
               
               // todo reflect / bounce according to BRDF
               
            }
         }
         num_bounces++;
      }

//            if (debug)
//               printf("%f %f %f\n", src.x, src.y, src.z);

      for (auto sample_index : src) {
         sample_sum += sample_index;
      }

//      if (debug)
//         printf("%f %f %f\n", sample_sum.x, sample_sum.y, sample_sum.z);
      
      sample_sum *= (255.f / samples);
//
//      if (debug)
//         printf("%f %f %f\n", sample_sum.x, sample_sum.y, sample_sum.z);
      
      device_pointers.device_samples->r[pixel_index] = sample_sum.x;
      device_pointers.device_samples->g[pixel_index] = sample_sum.y;
      device_pointers.device_samples->b[pixel_index] = sample_sum.z;
      
   }
   
   void PathTracerKernel::Trace() {

      struct device_pointers device_pointers {
         memory_manager->device_camera,
         memory_manager->meshes,
         memory_manager->num_meshes,
         memory_manager->device_samples
         
      };
      
      cuda_check_error( cudaMemcpyToSymbol(camera_to_world_matrix, memory_manager->camera_to_world_matrix, sizeof(float) * 16));
      
      constexpr unsigned int threadsPerBlock = 32;
      const unsigned int blocksPerGrid = (memory_manager->num_pixels + threadsPerBlock - 1) / threadsPerBlock;

      const unsigned int width = memory_manager->width;
      const unsigned int height = memory_manager->height;
      
      cudaError_t error = cudaSuccess;
      path_trace_kernel<<<blocksPerGrid, threadsPerBlock>>>(width, height, device_pointers);

      cudaDeviceSynchronize();
      error = cudaGetLastError();

      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to launch path_trace_kernel (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      //Intersection intersection = Scene->GetNearestShape(current_ray, x, y);
   }
}
