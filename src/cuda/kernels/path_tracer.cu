
// Created by daniel on 5/15/20.
//


#include <curand_kernel.h>
#include <cassert>
#include "path_tracer.cuh"
#include "common_device_functions.cuh"
#include "../check_error.h"


namespace poly {

   constexpr unsigned int threads_per_block = 32;
   
   __constant__ float camera_to_world_matrix[16];
   
   struct device_pointers {
      struct DeviceCamera* device_camera;
      
      struct DeviceMesh* device_meshes;
      unsigned int num_meshes;

      struct Samples* device_samples;
      
      poly::device_bvh_node* device_bvh_node;
      unsigned int num_bvh_nodes;
      
      poly::device_index_pair* device_index_pair;
      unsigned int num_index_pairs;
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
      float maxBoundFarT = poly::FloatMax;
      float minBoundNearT = 0;

      // TODO
      //const float gammaMultiplier = 1 + 2 * poly::Gamma(3);

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

   __device__ bool aabb_hits(const float* aabb, const float3 origin, const float3 inverse_direction) {

      //printf("bbox: (%f %f %f), (%f %f %f)\n", aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5], aabb[6]);

      //printf("o: (%f %f %f), d: (%f %f %f)\n", 
//             origin.x, origin.y, origin.y, inverse_direction.x, inverse_direction.y, inverse_direction.z);

      // TODO take ray's current t as a param and use for maxBoundFarT
      float maxBoundFarT = poly::Infinity;
      float minBoundNearT = 0;
      
      // TODO
      //const float gammaMultiplier = 1 + 2 * poly::Gamma(3);

      // X
      float tNear = (aabb[0] - origin.x) * inverse_direction.x;
      float tFar = (aabb[3] - origin.x) * inverse_direction.x;

      //printf("x tnear: %f, tfar: %f\n", tNear, tFar);
      
      float swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      //printf("x tnear: %f, tfar: %f\n", tNear, tFar);
      
      // TODO
      //tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;
      
      //printf("x min: %f, max: %f\n", minBoundNearT, maxBoundFarT);
      
      if (minBoundNearT > maxBoundFarT) {
         return false;
      }
      
      // Y
      tNear = (aabb[1] - origin.y) * inverse_direction.y;
      tFar = (aabb[4] - origin.y) * inverse_direction.y;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      // TODO
      // tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      // printf("y min: %f, max: %f\n", minBoundNearT, maxBoundFarT);
      
      if (minBoundNearT > maxBoundFarT) {
         return false;
      }

      // z
      tNear = (aabb[2] - origin.z) * inverse_direction.z;
      tFar = (aabb[5] - origin.z) * inverse_direction.z;

      //printf("z tnear: %f, tfar: %f\n", tNear, tFar);
      
      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      //printf("z tnear: %f, tfar: %f\n", tNear, tFar);
      
      // TODO
      // tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      //printf("z min: %f, max: %f\n", minBoundNearT, maxBoundFarT);
      
      return minBoundNearT <= maxBoundFarT;
   }

   __global__ void unit_test_ray_against_aabb_kernel(const float* aabb, float3 o, float3 id, int* result) {
      unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
      if (thread_index == 0)
         *result = aabb_hits(aabb, o, id) ? 1 : 0;
   }

   bool PathTracerKernel::unit_test_hit_ray_against_bounding_box(const poly::Ray &ray, const float* const device_aabb) {
      
      int* device_result;
      cuda_check_error( cudaMalloc((void**)&device_result, sizeof(int)) );
      
      float3 o = make_float3(ray.Origin.x, ray.Origin.y, ray.Origin.z);
      float3 id = {1.0f / ray.Direction.x, 1.0f / ray.Direction.y, 1.0f / ray.Direction.z};

      // invoke kernel that calls aabb_hits()
      unit_test_ray_against_aabb_kernel<<<1, 1>>>(device_aabb, o, id, device_result);
      
      // copy result from device

      int host_result = -1;
      cuda_check_error( cudaMemcpy(&host_result, device_result, sizeof(int), cudaMemcpyDeviceToHost) );
      
      // delete ray from device5
      cuda_check_error( cudaFree(device_result) );
      
      return (host_result == 1);
      
   }
   
   __device__ void consolidate_sample_intersections(
         struct device_pointers device_pointers,
         struct device_intersection* sample_intersections,
         const unsigned int num_samples,
         const float3* sample_origins,
         const float3* sample_directions
         ) {

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
   
   __device__ void linear_intersect(
         struct device_intersection* sample_intersections,
         const float3* const sample_origins,
         const float3* const sample_directions,
         const unsigned int num_samples,
         struct device_pointers device_pointers, 
         // TODO use something more efficient than bool array
         bool* active_mask,
         bool debug) {
      
      // const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;

      __shared__ float v0x[threads_per_block];
      __shared__ float v0y[threads_per_block];
      __shared__ float v0z[threads_per_block];

      __shared__ float v1x[threads_per_block];
      __shared__ float v1y[threads_per_block];
      __shared__ float v1z[threads_per_block];

      __shared__ float v2x[threads_per_block];
      __shared__ float v2y[threads_per_block];
      __shared__ float v2z[threads_per_block];

      debug = blockIdx.x == 6 && threadIdx.x == 17;
      
      if (debug) {
         printf("  intersection test:\n");
      }

      // for each mesh on the device
      for (unsigned int mesh_index = 0; mesh_index < device_pointers.num_meshes; mesh_index++) {
         DeviceMesh mesh = device_pointers.device_meshes[mesh_index];
         if (debug) {
            printf("    mesh_index %i\n", mesh_index);
            printf("      %p\n", &(device_pointers.device_meshes[mesh_index]));
            printf("      aabb (%f, %f, %f), (%f, %f, %f)\n", mesh.aabb[0], mesh.aabb[1], mesh.aabb[2], mesh.aabb[3], mesh.aabb[4], mesh.aabb[5]);
            printf("      num_faces %i\n", mesh.num_faces);
            printf("      num_vertices %i\n", mesh.num_vertices);
            printf("      num_bytes %i\n", mesh.num_bytes);
         }
         bool all_samples_miss = true;
         
         // check to see if all samples miss the mesh's AABB
         for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {
            bool hits_aabb = aabb_hits(mesh, sample_origins[sample_index], sample_directions[sample_index]);
            if (hits_aabb)
               all_samples_miss = false;
         }

         // if all samples miss, we can safely skip the mesh
         if (__all_sync(0xFFFFFFFF, all_samples_miss)) {
            if (debug) {
               printf("      all samples in the warp missed the mesh's aabb\n");   
            }
            continue;
         }
         
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
               if (debug) {
                  printf("      hit face %i\n", face_index);
                  printf("      t %f\n", ft);
               }
            }
         }
      }

      consolidate_sample_intersections(
            device_pointers,
            sample_intersections,
            num_samples,
            sample_origins,
            sample_directions
      );
   }
   
   __device__ void bvh_intersect(
         struct device_intersection* sample_intersections,
         const float3* const sample_origins,
         const float3* const sample_directions,
         const float3* const sample_inverse_directions,
         const unsigned int num_samples,
         bool* active_mask,
         const unsigned int bounce_num,
         struct device_pointers device_pointers) {
      
      
      // TODO - figure out a bound on how big this could/should be
      int future_node_stack[1024];
      
      /**
       * first free index in future_node_stack
       */
      int next_future_node_index = 0;
      future_node_stack[next_future_node_index] = 0;

      bool debug = blockIdx.x == 6 && threadIdx.x == 17 && bounce_num == 1;
      
      poly::device_bvh_node node;
      int current_node_index = 0;
         
      do {
         node = device_pointers.device_bvh_node[current_node_index];
         const bool is_leaf = node.is_leaf();
         
//         if (debug) {
//            printf("Loaded node %i\n", current_node_index);
//         }
         
         if (aabb_hits(node.aabb, sample_origins[0], sample_inverse_directions[0])) {
            if (is_leaf) {
               for (unsigned int i = 0; i < node.num_faces; i++) {
                  
                  device_index_pair indices = device_pointers.device_index_pair[node.offset + i];
                  const unsigned int mesh_index = indices.mesh_index;
                  const unsigned int face_index = indices.face_index;
                  
                  if (debug) {
                     printf("Tested face_index %i\n", indices.face_index);
                  }

//                  <<<6, 17>>> bounce 1 hit mismatch: bvh: 0, linear: 1
//                  <<<6, 17>>> face_index: bvh: 0, linear: 43582
//                  <<<6, 17>>> t: bvh: inf, linear: 0.039489

                  if (debug && face_index == 43582) {
                     printf("Loaded face_index %i\n", face_index);
                  } 
                  
                  const DeviceMesh mesh = device_pointers.device_meshes[mesh_index];
                  
                  // TODO intersect face
                  const float3 v0 = make_float3(mesh.x[face_index], mesh.y[face_index], mesh.z[face_index]);
                  const float3 v1 = make_float3(mesh.x[face_index + (mesh.num_faces)], mesh.y[face_index + (mesh.num_faces)], mesh.z[face_index + (mesh.num_faces)]);
                  const float3 v2 = make_float3(mesh.x[face_index + (mesh.num_faces * 2)], mesh.y[face_index + (mesh.num_faces * 2)], mesh.z[face_index + (mesh.num_faces * 2)]);

                  const float3 e0 = v1 - v0;
                  const float3 e1 = v2 - v1;
                  float3 pn = cross(e0, e1);

                  pn *= __frsqrt_rn(dot(pn, pn));
                  for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {
                     // don't intersect a sample if it's not active
                     
                     if (!active_mask[sample_index]) {
                        if (debug) {
                           printf("Bailing on sample %i for face_index %i because lane is not active\n", sample_index, face_index);
                        }
                        continue;
                     }

                     float ft = INFINITY;
                     float3 ro = sample_origins[sample_index];
                     float3 rd = sample_directions[sample_index];

                     device_intersection& intersection = sample_intersections[sample_index];
                     {
                        const float divisor = dot(pn, rd);
                        if (divisor == 0.0f) {
                           if (debug) {
                              printf("Bailing on face_index %i because ray is parallel to face\n", sample_index, face_index);
                           }
                           // parallel
                           continue;
                        }

                        ft = (dot(pn, v0 - ro)) / divisor;
                     }

                     if (ft <= 0 || ft > intersection.t) {
                        if (debug) {
                           printf("Bailing on face_index %i due to previous t %i vs this t %i \n", face_index, intersection.t, ft);
                        }
                        continue;
                     }

                     const float3 hp = ro + rd * ft;

                     {
                        float3 p = hp - v0;
                        float3 cross_ep = cross(e0, p);
                        float normal = dot(cross_ep, pn);
                        if (normal <= 0) {
                           if (debug) {
                              printf("Bailing on face_index %i due to first normal test %f\n", face_index, normal);
                           }
                           continue;
                        }

                        p = hp - v1;
                        cross_ep = cross(e1, p);
                        normal = dot(cross_ep, pn);
                        if (normal <= 0) {
                           if (debug) {
                              printf("Bailing on face_index %i due to second normal test %f\n", face_index, normal);
                           }
                           continue;
                        }

                        const float3 e2 = v0 - v2;
                        p = hp - v2;
                        cross_ep = cross(e2, p);
                        normal = dot(cross_ep, pn);
                        if (normal <= 0) {
                           if (debug) {
                              printf("Bailing on face_index %i due to third normal test %f\n", face_index, normal);
                           }
                           continue;
                        }
                     }

                     
                     
                     // temp
                     intersection.hits = true;
                     intersection.t = ft;
                     intersection.face_index = face_index;
                     intersection.mesh_index = mesh_index;
                  }
               }
               
               // next node should be from the stack, if any
               if (next_future_node_index == 0) {
                  break;
               }
               current_node_index = future_node_stack[--next_future_node_index];
            }
            else {
               // next node should be one of the two children, and
               // push the other node on the stack
               // TODO push closer node first
               future_node_stack[next_future_node_index++] = current_node_index + node.offset;
               current_node_index = current_node_index + 1;
            }
         }
         else {
            // next node should be from the stack
            if (next_future_node_index == 0) {
               break;
            }
            current_node_index = future_node_stack[--next_future_node_index];
         }
      } while (current_node_index >= 0);
      
      consolidate_sample_intersections(
            device_pointers,
            sample_intersections,
            num_samples,
            sample_origins,
            sample_directions
            );
   }
   
   __global__ void path_trace_kernel(const unsigned int width, const unsigned int height, const float tan_fov_half, struct device_pointers device_pointers) {
      // loop over pixels
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      const unsigned int pixel_x = pixel_index % width;
      const unsigned int pixel_y = pixel_index / width;

      // generate camera rays
      constexpr unsigned int samples = 1 * 1;
      
      const float aspect = (float) width / (float) height;

      const float3 camera_origin = {0, 0, 0};
      const float3 world_origin_orig = matrix_apply_point(camera_to_world_matrix, camera_origin);
      
      // samples
      const int root_samples = ceil(sqrtf(samples));
      const float offset_step = 1.f / (float)root_samples;
      const float offset_step_initial = offset_step / 2.f;

      bool debug = false;
      if (pixel_x == 275 && pixel_y == 275) {
         debug = true;
         printf("debug\n");
      }

      curandState state;
      curand_init(pixel_index, pixel_index, 0, &state);
      
      float3 sample_origins[samples];
      float3 sample_directions[samples];
      float3 sample_directions_inverse[samples];
      
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
            
            sample_directions_inverse[sample_index] = {
                  1.f / world_direction.x,
                  1.f / world_direction.y,
                  1.f / world_direction.z
            };
         }
      }

      float3 src[samples];
      bool active_linear[samples];
      bool active_bvh[samples];
      for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
         // TODO initialize
         src[sample_index] = { 1.0f, 1.0f, 1.0f};
         active_linear[sample_index] = true;
         active_bvh[sample_index] = true;
      }

      device_intersection sample_intersections_bvh[samples];
      device_intersection sample_intersections_linear[samples];
      unsigned int num_bounces = 0;
      while (true) {
         if (debug) {
            printf("thread %i: bounce %i\n", threadIdx.x, num_bounces);
         }
         
         if (num_bounces > 5) {
            // kill any samples that are still active
            for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
               if (active_linear[sample_index]) {
                  src[sample_index] = {0, 0, 0};
                  if (debug) {
                     printf("  sample %i: killed\n", sample_index);
                  }
               }
            }
            break;
         }
         
         // if no samples are active in this warp, we're done
         bool any_sample_active = false;
         for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
            
            if (active_linear[sample_index] || active_bvh[sample_index])
               any_sample_active = true;
         }
         
         if (__all_sync(0xFFFFFFFF, !any_sample_active)) {
            if (debug) {
               printf("  exited loop before intersecting bounce %i (all samples in warp inactive)\n", num_bounces);
            }
            break;
         }

         //printf("  didn't exit loop before intersecting bounce %i (some samples in warp active)\n", num_bounces);
         
         // reset all intersections
         for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
            sample_intersections_bvh[sample_index].face_index = 0;
            sample_intersections_bvh[sample_index].mesh_index = 0;
            sample_intersections_bvh[sample_index].hit_point = {0.f, 0.f, 0.f };
            sample_intersections_bvh[sample_index].t = INFINITY;
            sample_intersections_bvh[sample_index].hits = false;

            sample_intersections_linear[sample_index].face_index = 0;
            sample_intersections_linear[sample_index].mesh_index = 0;
            sample_intersections_linear[sample_index].hit_point = {0.f, 0.f, 0.f };
            sample_intersections_linear[sample_index].t = INFINITY;
            sample_intersections_linear[sample_index].hits = false;
         }
         
         if (1) {
            linear_intersect(
                  sample_intersections_linear,
                  sample_origins,
                  sample_directions,
                  samples,
                  device_pointers,
                  active_linear,
                  debug);
         }
         else {
            bvh_intersect(
                  sample_intersections_bvh,
                  sample_origins,
                  sample_directions,
                  sample_directions_inverse,
                  samples,
                  active_bvh,
                  num_bounces,
                  device_pointers);
         }

//         if (sample_intersections_bvh->hits != sample_intersections_linear->hits) {
//            printf("<<<%i, %i>>> bounce %i hit mismatch: bvh: %i, linear: %i\n<<<%i, %i>>> face_index: bvh: %i, linear: %i\n<<<%i, %i>>> t: bvh: %f, linear: %f\n", blockIdx.x, threadIdx.x, num_bounces, sample_intersections_bvh->hits, sample_intersections_linear->hits, blockIdx.x, threadIdx.x, sample_intersections_bvh->face_index, sample_intersections_linear->face_index, blockIdx.x, threadIdx.x, sample_intersections_bvh->t, sample_intersections_linear->t);
////            __syncthreads();
////            //printf("mesh_index: bvh: %i, linear: %i\n", sample_intersections_bvh->mesh_index, sample_intersections_linear->mesh_index);
////            printf("<<<%i, %i>>> face_index: bvh: %i, linear: %i\n", blockIdx.x, threadIdx.x, sample_intersections_bvh->face_index, sample_intersections_linear->face_index);
////            __syncthreads();
////            printf("<<<%i, %i>>> t: bvh: %f, linear: %f\n", blockIdx.x, threadIdx.x, sample_intersections_bvh->t, sample_intersections_linear->t);
////            __syncthreads();
//         }
//         else if (sample_intersections_bvh->hits) {
//            if (sample_intersections_bvh->mesh_index != sample_intersections_linear->mesh_index) {
//               printf("mesh_index mismatch: bvh: %i, linear: %i\n", sample_intersections_bvh->mesh_index, sample_intersections_linear->mesh_index);
//            }
//            if (sample_intersections_bvh->face_index != sample_intersections_linear->face_index) {
//               printf("face_index mismatch: bvh: %i, linear: %i\n", sample_intersections_bvh->face_index, sample_intersections_linear->face_index);
//            }
//            if (sample_intersections_bvh->t != sample_intersections_linear->t) {
//               printf("t mismatch: bvh: %i, linear: %i\n", sample_intersections_bvh->t, sample_intersections_linear->t);
//            }
//         }

         //printf("finished bvh\n");
         
         // process each sample
         for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
            if (debug) {
               printf("  sample %i: \n", sample_index);
            }
            if (active_linear[sample_index]) {
               device_intersection intersection = sample_intersections_bvh[sample_index];
               float3 ro = sample_origins[sample_index];
               float3 rd = sample_directions[sample_index];
               
               if (!intersection.hits) {
                  if (debug) {
                     printf("    no hit, becoming inactive\n");
//                     printf("    o: %f %f %f\n", ro.x, ro.y, ro.z);
//                     printf("    d: %f %f %f\n", rd.x, rd.y, rd.z);
                  }
                  active_linear[sample_index] = false;
                  continue;
               }

               if (debug) {
                  printf("    hit mesh_index %i face_index %i\n", intersection.mesh_index, intersection.face_index);
                  printf("    o: %f %f %f\n", ro.x, ro.y, ro.z);
                  printf("    d: %f %f %f\n", rd.x, rd.y, rd.z);
               }

               // todo hit light

               // TODO put this into a lambert_brdf function
//                const poly::Vector local_incoming = intersection.WorldToLocal(current_ray.Direction);
               float3 local_incoming = normalize(make_float3(dot(rd, intersection.tangent1), dot(rd, intersection.normal), dot(rd, intersection.tangent2)));

               // mirror
               const float3 local_outgoing = make_float3(local_incoming.x, -local_incoming.y, local_incoming.z); 

               // lambert
               float u0 = curand_uniform(&state);
               float u1 = curand_uniform(&state);
//               float3 local_outgoing = cosine_sample_hemisphere(u0, u1);

               // const poly::Vector world_outgoing = intersection.LocalToWorld(local_outgoing);
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
               src[sample_index] = src[sample_index] * make_float3(0.8f, 0.5f, 0.5f);

               if (debug) {
                  printf("    bounce ray:\n");
                  printf("    o: %f %f %f\n", sample_origins[sample_index].x, sample_origins[sample_index].y, sample_origins[sample_index].z);
                  printf("    d: %f %f %f\n", sample_directions[sample_index].x, sample_directions[sample_index].y, sample_directions[sample_index].z);
               }
               
               // todo reflect / bounce according to BRDF
               
            }
            else {
               if (debug) {
                  printf("    inactive\n");
               }
            }
         }
         num_bounces++;
      }

      float3 sample_sum = make_float3(0, 0, 0);
      for (unsigned int sample_index = 0; sample_index < samples; sample_index++) {
         sample_sum += src[sample_index];
      }
      sample_sum *= (255.f / samples);
      device_pointers.device_samples->r[pixel_index] = sample_sum.x;
      device_pointers.device_samples->g[pixel_index] = sample_sum.y;
      device_pointers.device_samples->b[pixel_index] = sample_sum.z;
   }
   
   __global__ void box_filter_kernel(float3* src, struct Samples* device_samples, const unsigned int num_samples) {
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      float3 sample_sum = make_float3(0, 0, 0);

      for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {

         float3 sample = src[sample_index];
         sample_sum += src[sample_index];
      }

      sample_sum *= (255.f / num_samples);

      device_samples->r[pixel_index] = sample_sum.x;
      device_samples->g[pixel_index] = sample_sum.y;
      device_samples->b[pixel_index] = sample_sum.z;
   }
   
   void PathTracerKernel::Trace() const {

      struct device_pointers device_pointers {
         memory_manager->device_camera,
         memory_manager->meshes,
         memory_manager->num_meshes,
         memory_manager->device_samples,
         memory_manager->device_bvh,
         memory_manager->num_bvh_nodes,
         memory_manager->index_pair,
         memory_manager->num_indices
      };

      const float tan_fov_half = std::tan(memory_manager->camera_fov * poly::PIOver360);
      
      cuda_check_error( cudaMemcpyToSymbol(camera_to_world_matrix, memory_manager->camera_to_world_matrix, sizeof(float) * 16));
      
      const unsigned int blocksPerGrid = (memory_manager->num_pixels + threads_per_block - 1) / threads_per_block;

      const unsigned int width = memory_manager->width;
      const unsigned int height = memory_manager->height;
      
      cudaError_t error = cudaSuccess;
      printf("launching with <<<%i, %i>>>\n", blocksPerGrid, threads_per_block);
      path_trace_kernel<<<blocksPerGrid, threads_per_block>>>(width, height, tan_fov_half, device_pointers);

      cudaDeviceSynchronize();
      error = cudaGetLastError();

      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to launch path_trace_kernel (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
   }
}
