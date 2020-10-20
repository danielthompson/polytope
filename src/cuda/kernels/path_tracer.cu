
// Created by daniel on 5/15/20.
//


#include <curand_kernel.h>
#include <cassert>
#include <cmath>
#include "path_tracer.cuh"
#include "common_device_functions.cuh"
#include "../check_error.h"


namespace poly {

   constexpr unsigned int threads_per_block = 32;
   
   __constant__ float camera_to_world_matrix[16];
   
   struct device_pointers {
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

   __device__ bool aabb_hits(const float* aabb, const float3 origin, const float3 inverse_direction, float* const mint) {

      // TODO take ray's current t as a param and use for maxBoundFarT
      float maxBoundFarT = poly::Infinity;
      float minBoundNearT = 0;
      
      // TODO
      //const float gammaMultiplier = 1 + 2 * poly::Gamma(3);

      // X
      float tNear = (aabb[0] - origin.x) * inverse_direction.x;
      float tFar = (aabb[3] - origin.x) * inverse_direction.x;

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
      tNear = (aabb[1] - origin.y) * inverse_direction.y;
      tFar = (aabb[4] - origin.y) * inverse_direction.y;

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
      tNear = (aabb[2] - origin.z) * inverse_direction.z;
      tFar = (aabb[5] - origin.z) * inverse_direction.z;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      // TODO
      // tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      *mint = minBoundNearT;
      
      return minBoundNearT <= maxBoundFarT;
   }

   __global__ void unit_test_ray_against_aabb_kernel(const float* aabb, float3 o, float3 id, int* result) {
      unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
      float t = 0;
      if (thread_index == 0)
         *result = aabb_hits(aabb, o, id, &t) ? 1 : 0;
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
         struct device_intersection* intersection,
         const float3 origin,
         const float3 direction
         ) {

      if (intersection->hits) {

         // calculate normal at hit point
         DeviceMesh mesh_hit = device_pointers.device_meshes[intersection->mesh_index];

         const unsigned int v1_index = intersection->face_index + mesh_hit.num_faces;
         const unsigned int v2_index = intersection->face_index + mesh_hit.num_faces * 2;

         const float3 v0 = {device_pointers.device_meshes[intersection->mesh_index].x[intersection->face_index],
                            device_pointers.device_meshes[intersection->mesh_index].y[intersection->face_index],
                            device_pointers.device_meshes[intersection->mesh_index].z[intersection->face_index]};
         const float3 v1 = {device_pointers.device_meshes[intersection->mesh_index].x[v1_index],
                            device_pointers.device_meshes[intersection->mesh_index].y[v1_index],
                            device_pointers.device_meshes[intersection->mesh_index].z[v1_index]};
         const float3 v2 = {device_pointers.device_meshes[intersection->mesh_index].x[v2_index],
                            device_pointers.device_meshes[intersection->mesh_index].y[v2_index],
                            device_pointers.device_meshes[intersection->mesh_index].z[v2_index]};

         const float3 e0 = v1 - v0;
         const float3 e1 = v2 - v1;
         const float3 e2 = v0 - v2;
         float3 n = cross(e0, e1);

         // flip normal if needed
         const float ray_dot_normal = dot(direction, n);
         const float flip_factor = ray_dot_normal > 0 ? -1 : 1;
         n *= flip_factor;

         normalize(n);
         intersection->normal = n;

         // TODO this is (very) non-optimal
         intersection->hit_point = make_float3(
               fma(direction.x, intersection->t, origin.x),
               fma(direction.y, intersection->t, origin.y),
               fma(direction.z, intersection->t, origin.z)
         );

         // offset origin - hack - temp
         intersection->hit_point.x = fma(n.x, 0.002f, intersection->hit_point.x);
         intersection->hit_point.y = fma(n.y, 0.002f, intersection->hit_point.y);
         intersection->hit_point.z = fma(n.z, 0.002f, intersection->hit_point.z);

         const float edge0dot = abs(dot(e0, e1));
         const float edge1dot = abs(dot(e1, e2));
         const float edge2dot = abs(dot(e2, e0));

         if (edge0dot > edge1dot && edge0dot > edge2dot) {
            intersection->tangent1 = e0;
         } else if (edge1dot > edge0dot && edge1dot > edge2dot) {
            intersection->tangent1 = e1;
         } else {
            intersection->tangent1 = e2;
         }

         intersection->tangent2 = cross(intersection->tangent1, n);

         normalize(intersection->tangent1);
         normalize(intersection->tangent2);
      }
      
   }
   
//   __device__ void linear_intersect(
//         struct device_intersection* sample_intersections,
//         const float3* const sample_origins,
//         const float3* const sample_directions,
//         const unsigned int num_samples,
//         struct device_pointers device_pointers, 
//         // TODO use something more efficient than bool array
//         bool* active_mask,
//         bool debug) {
//      
//      // const unsigned int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
//
//      __shared__ float v0x[threads_per_block];
//      __shared__ float v0y[threads_per_block];
//      __shared__ float v0z[threads_per_block];
//
//      __shared__ float v1x[threads_per_block];
//      __shared__ float v1y[threads_per_block];
//      __shared__ float v1z[threads_per_block];
//
//      __shared__ float v2x[threads_per_block];
//      __shared__ float v2y[threads_per_block];
//      __shared__ float v2z[threads_per_block];
//
//      debug = blockIdx.x == 6 && threadIdx.x == 17;
//      
//      if (debug) {
//         printf("  intersection test:\n");
//      }
//
//      // for each mesh on the device
//      for (unsigned int mesh_index = 0; mesh_index < device_pointers.num_meshes; mesh_index++) {
//         DeviceMesh mesh = device_pointers.device_meshes[mesh_index];
//         if (debug) {
//            printf("    mesh_index %i\n", mesh_index);
//            printf("      %p\n", &(device_pointers.device_meshes[mesh_index]));
//            printf("      aabb (%f, %f, %f), (%f, %f, %f)\n", mesh.aabb[0], mesh.aabb[1], mesh.aabb[2], mesh.aabb[3], mesh.aabb[4], mesh.aabb[5]);
//            printf("      num_faces %i\n", mesh.num_faces);
//            printf("      num_vertices %i\n", mesh.num_vertices);
//            printf("      num_bytes %i\n", mesh.num_bytes);
//         }
//         bool all_samples_miss = true;
//         
//         // check to see if all samples miss the mesh's AABB
//         for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {
//            bool hits_aabb = aabb_hits(mesh, sample_origins[sample_index], sample_directions[sample_index]);
//            if (hits_aabb)
//               all_samples_miss = false;
//         }
//
//         // if all samples miss, we can safely skip the mesh
//         if (__all_sync(0xFFFFFFFF, all_samples_miss)) {
//            if (debug) {
//               printf("      all samples in the warp missed the mesh's aabb\n");   
//            }
//            continue;
//         }
//         
//         const float* mesh_x = mesh.x;
//         const float* mesh_y = mesh.y;
//         const float* mesh_z = mesh.z;
//         
//         // for each face in the mesh
//         for (unsigned int face_index = 0; face_index < mesh.num_faces; face_index++) {
//            // every 32 faces, have the threads in the warp coordinate to load the next 32 faces' vertices from global to shared memory
//            const unsigned int shared_index = face_index & 0x0000001Fu;
//            const unsigned int thread_face_index = face_index + threadIdx.x;
//            if (shared_index == 0 && thread_face_index < mesh.num_faces) {
//               v0x[threadIdx.x] = mesh_x[thread_face_index];
//               v0y[threadIdx.x] = mesh_y[thread_face_index];
//               v0z[threadIdx.x] = mesh_z[thread_face_index];
//               
//               const unsigned int v1_index = thread_face_index + mesh.num_faces;
//               v1x[threadIdx.x] = mesh_x[v1_index];
//               v1y[threadIdx.x] = mesh_y[v1_index];
//               v1z[threadIdx.x] = mesh_z[v1_index];
//
//               const unsigned int v2_index = thread_face_index + (mesh.num_faces * 2);
//               v2x[threadIdx.x] = mesh_x[v2_index];
//               v2y[threadIdx.x] = mesh_y[v2_index];
//               v2z[threadIdx.x] = mesh_z[v2_index];
//            }
//            
//            __syncthreads();
//            
////            assert(v0x[shared_index] == mesh.x[face_index]);
////            assert(v0y[face_index % 32] == mesh.y[face_index]);
////            assert(v0z[face_index % 32] == mesh.z[face_index]);
////
////            assert(v1x[face_index % 32] == mesh.x[face_index + v1_index_offset]);
////            assert(v1y[face_index % 32] == mesh.y[face_index + v1_index_offset]);
////            assert(v1z[face_index % 32] == mesh.z[face_index + v1_index_offset]);
////        
////            assert(v2x[face_index % 32] == mesh.x[face_index + v2_index_offset]);
////            assert(v2y[face_index % 32] == mesh.y[face_index + v2_index_offset]);
////            assert(v2z[face_index % 32] == mesh.z[face_index + v2_index_offset]);
//
//            const float3 v0 = make_float3(v0x[shared_index], v0y[shared_index], v0z[shared_index]);
//            //const unsigned int v1index = face_index + v1_index_offset;
//            const float3 v1 = {v1x[shared_index], v1y[shared_index], v1z[shared_index]};
//            const float3 e0 = v1 - v0;
//            //const unsigned int v2index = face_index + v2_index_offset;
//            const float3 v2 = {v2x[shared_index], v2y[shared_index], v2z[shared_index]};
//            const float3 e1 = v2 - v1;
//            float3 pn = cross(e0, e1);
//
//            //const float oneOverLength = 1.0f / sqrt(dot(pn, pn));
//            pn *= __frsqrt_rn(dot(pn, pn));
////            pn *= (1.0f / sqrtf(dot(pn, pn)));
//            
//            // TODO do all samples for a vertex batch at once, since loading the vertices from global to shared memory is the bottleneck
//            for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {
//               // don't intersect a sample if it's not active
//               if (!active_mask[sample_index])
//                  continue;
//
//               float ft = INFINITY;
//               float3 ro = sample_origins[sample_index];
//               float3 rd = sample_directions[sample_index];
////               if (debug) {
////                  printf("intersecting:\n");
////                  printf("o: %f %f %f\n", ro.x, ro.y, ro.z);
////                  printf("d: %f %f %f\n", rd.x, rd.y, rd.z);
////               }
//               
//               device_intersection& intersection = sample_intersections[sample_index];
//               {
//                  const float divisor = dot(pn, rd);
//                  if (divisor == 0.0f) {
//                     // parallel
//                     continue;
//                  }
//
//                  ft = (dot(pn, v0 - ro)) / divisor;
//               }
//
//               if (ft <= 0 || ft > intersection.t) {
//                  continue;
//               }
//
////            if (face_index == 1068 && debug) {
////               printf("t: %f\n", ft);
////            }
//
//               const float3 hp = ro + rd * ft;
//
//               {
//                  float3 p = hp - v0;
//                  float3 cross_ep = cross(e0, p);
//                  float normal = dot(cross_ep, pn);
//                  if (normal <= 0)
//                     continue;
//
//                  p = hp - v1;
//                  cross_ep = cross(e1, p);
//                  normal = dot(cross_ep, pn);
//                  if (normal <= 0)
//                     continue;
//
//                  const float3 e2 = v0 - v2;
//                  p = hp - v2;
//                  cross_ep = cross(e2, p);
//                  normal = dot(cross_ep, pn);
//                  if (normal <= 0)
//                     continue;
//               }
//
//               // temp
//               intersection.hits = true;
//               intersection.t = ft;
//               intersection.face_index = face_index;
//               intersection.mesh_index = mesh_index;
//               if (debug) {
//                  printf("      hit face %i\n", face_index);
//                  printf("      t %f\n", ft);
//               }
//            }
//         }
//      }
//
//      consolidate_sample_intersections(
//            device_pointers,
//            sample_intersections,
//            num_samples,
//            sample_origins,
//            sample_directions
//      );
//   }
//   
   __device__ void bvh_intersect(
         struct device_intersection* intersection,
         const float3 origin,
         const float3 direction,
         const float3 direction_inverse,
         const unsigned int bounce_num,
         struct device_pointers device_pointers) {
      
      // TODO - figure out a bound on how big this could/should be
      int future_node_stack[1024];
      
      /**
       * first free index in future_node_stack
       */
      int next_future_node_index = 0;
      future_node_stack[next_future_node_index] = 0;

      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      bool debug = pixel_index == 131328;
      
      poly::device_bvh_node node;
      int current_node_index = 0;
         
      do {
         node = device_pointers.device_bvh_node[current_node_index];
         if (debug) {
            printf("<<<%i, %i>>> Looking at node %i\n", blockIdx.x, threadIdx.x, current_node_index);
         }
         const bool is_leaf = node.is_leaf();


         if (debug) {
            printf("<<<%i, %i>>> intersecting node %i with ray o (%f, %f, %f) d (%f %f %f), id (%f %f %f)\n", 
                   blockIdx.x, threadIdx.x, current_node_index,
                   origin.x, origin.y, origin.z,
                   direction.x, direction.y, direction.z,
                   direction_inverse.x, direction_inverse.y, direction_inverse.z);
         }
         
         float aabb_t = INFINITY;
         
         if (aabb_hits(node.aabb, origin, direction_inverse, &aabb_t) && aabb_t < intersection->t) {
            
            if (debug) {
               printf("<<<%i, %i>>> ray hit node %i aabb\n", blockIdx.x, threadIdx.x, current_node_index);
            }
            if (is_leaf) {
               if (debug) {
                  printf("<<<%i, %i>>> node %i is leaf, intersecting faces\n", blockIdx.x, threadIdx.x,
                         current_node_index);
               }
               for (int i = 0; i < node.num_faces; i++) {

                  device_index_pair indices = device_pointers.device_index_pair[node.offset + i];
                  const unsigned int mesh_index = indices.mesh_index;
                  const unsigned int face_index = indices.face_index;

                  if (debug) {
                     printf("Tested face_index %i\n", indices.face_index);
                  }

                  const DeviceMesh mesh = device_pointers.device_meshes[mesh_index];

                  // TODO intersect face
                  const float3 v0 = make_float3(mesh.x[face_index], mesh.y[face_index], mesh.z[face_index]);
                  const float3 v1 = make_float3(mesh.x[face_index + (mesh.num_faces)],
                                                mesh.y[face_index + (mesh.num_faces)],
                                                mesh.z[face_index + (mesh.num_faces)]);
                  const float3 v2 = make_float3(mesh.x[face_index + (mesh.num_faces * 2)],
                                                mesh.y[face_index + (mesh.num_faces * 2)],
                                                mesh.z[face_index + (mesh.num_faces * 2)]);

                  const float3 e0 = v1 - v0;
                  const float3 e1 = v2 - v1;
                  float3 pn = cross(e0, e1);

                  pn *= __frsqrt_rn(dot(pn, pn));
                  float ft = INFINITY;
                  
                  {
                     const float divisor = dot(pn, direction);
                     if (divisor == 0.0f) {
                        if (debug) {
                           printf("<<<%i, %i>>> ray is parallel to face %i, bailing\n", blockIdx.x, threadIdx.x,
                                  face_index);
                        }
                        // parallel
                        continue;
                     }

                     ft = (dot(pn, v0 - origin)) / divisor;
                  }

                  if (ft <= 0 || ft > intersection->t) {
                     if (debug) {
                        printf("Bailing on face_index %i due to previous t %i vs this t %i \n", face_index,
                               intersection->t, ft);
                     }
                     continue;
                  }

                  const float3 hp = origin + direction * ft;

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


                  if (debug) {
                     printf("Hit face face_index %i with t %f\n", face_index, ft);
                  }
                  // temp
                  intersection->hits = true;
                  intersection->t = ft;
                  intersection->face_index = face_index;
                  intersection->mesh_index = mesh_index;
               }
               

               // next node should be from the stack, if any
               if (next_future_node_index == 0) {
                  if (debug) {
                     printf("<<<%i, %i>>> no more nodes on stack, ending traversal\n", blockIdx.x, threadIdx.x,
                            current_node_index);
                  }
                  break;
               }
               if (debug) {
                  printf("<<<%i, %i>>> picking up next node from stack\n", blockIdx.x, threadIdx.x,
                         current_node_index);
               }
               current_node_index = future_node_stack[--next_future_node_index];
            } else {
               if (debug) {
                  printf("<<<%i, %i>>> node %i is interior, pushing high child on to stack, looking at low child\n",
                         blockIdx.x, threadIdx.x, current_node_index);
               }
               // next node should be one of the two children, and
               // push the other node on the stack
               const int high_child_index = current_node_index + 1;
               const int low_child_index = current_node_index + node.offset;
               
               // visit closer node first, push farther node onto stack
               if ((node.flags == 0 && direction.x > 0)
                  || (node.flags == 1 && direction.y > 0)
                  || (node.flags == 2 && direction.z > 0)) {
                  current_node_index = high_child_index;
                  future_node_stack[next_future_node_index++] = low_child_index;
               }
               else {
                  current_node_index = low_child_index;
                  future_node_stack[next_future_node_index++] = high_child_index;
               }
            }

         }
         else {
            if (debug) {
               printf("<<<%i, %i>>> ray hit node %i aabb\n", blockIdx.x, threadIdx.x, current_node_index);
            }
               // next node should be from the stack
            if (next_future_node_index == 0) {
               if (debug) {
                  printf("<<<%i, %i>>> no more nodes on stack, ending traversal with hit %i and t %f\n", blockIdx.x, threadIdx.x, current_node_index, intersection->hits, intersection->t);
               }
               break;
            }
            if (debug) {
               printf("<<<%i, %i>>> picking up next node from stack\n", blockIdx.x, threadIdx.x, current_node_index);
            }
            current_node_index = future_node_stack[--next_future_node_index];
         }
      } while (current_node_index >= 0);
      
      consolidate_sample_intersections(
            device_pointers,
            intersection,
            origin,
            direction
            );
   }
   
   __global__ void path_trace_kernel(
         float3* const sample_origins,
         float3* const sample_directions,
         float3* const sample_directions_inverse,
         const unsigned int width, 
         const unsigned int height, 
         struct device_pointers device_pointers) {
      // loop over pixels
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;

      bool debug = pixel_index == 131328;

      float3 src = { 1.0f, 1.0f, 1.0f};
      
      bool active = true;
      
      device_intersection intersection;
      
      float3 origin = sample_origins[pixel_index];
      float3 direction = sample_directions[pixel_index];
      float3 direction_inverse = sample_directions_inverse[pixel_index];
      
      unsigned int num_bounces = 0;
      while (true) {
         if (debug) {
            printf("<<<%i, %i>>> bounce %i\n", blockIdx.x, threadIdx.x, num_bounces);
         }
         
         if (num_bounces > 5) {
            break;
         }
         
         // reset all intersections
         intersection.face_index = 0;
         intersection.mesh_index = 0;
         intersection.hit_point = {0.f, 0.f, 0.f };
         intersection.t = INFINITY;
         intersection.hits = false;
         bvh_intersect(
            &intersection,
            origin,
            direction,
            direction_inverse,
            num_bounces,
            device_pointers);
         
         
         // process each sample
         if (active) {
            if (intersection.hits) {
               if (debug) {
                  printf("    hit mesh_index %i face_index %i\n", intersection.mesh_index, intersection.face_index);
                  printf("    o: %f %f %f\n", origin.x, origin.y, origin.z);
                  printf("    d: %f %f %f\n", direction.x, direction.y, direction.z);
               }

               // todo hit light

               // TODO put this into a lambert_brdf function
//                const poly::Vector local_incoming = intersection.WorldToLocal(current_ray.Direction);
               float3 local_incoming = normalize(
                     make_float3(dot(direction, intersection.tangent1), 
                                 dot(direction, intersection.normal),
                                 dot(direction, intersection.tangent2)));

               // mirror
               const float3 local_outgoing = make_float3(local_incoming.x, -local_incoming.y, local_incoming.z);

               // lambert
               // TODO refactor curand initialization
//               float u0 = curand_uniform(&state);
//               float u1 = curand_uniform(&state);
////               float3 local_outgoing = cosine_sample_hemisphere(u0, u1);

               // const poly::Vector world_outgoing = intersection.LocalToWorld(local_outgoing);
               float3 world_outgoing = normalize(make_float3(
                     intersection.tangent1.x * local_outgoing.x + intersection.normal.x * local_outgoing.y +
                     intersection.tangent2.x * local_outgoing.z,
                     intersection.tangent1.y * local_outgoing.x + intersection.normal.y * local_outgoing.y +
                     intersection.tangent2.y * local_outgoing.z,
                     intersection.tangent1.z * local_outgoing.x + intersection.normal.z * local_outgoing.y +
                     intersection.tangent2.z * local_outgoing.z
               ));
//               if (debug) {
//                  printf("n: %f %f %f\n", intersection.normal.x, intersection.normal.y, intersection.normal.z);
//                  printf("o: %f %f %f\n", intersection.hit_point.x, intersection.hit_point.y, intersection.hit_point.z);
//                  printf("d: %f %f %f\n", world_outgoing.x, world_outgoing.y, world_outgoing.z);
//               }

               // TOOD offset
               origin = intersection.hit_point;
               direction = world_outgoing;
               direction_inverse = make_float3(1.f / world_outgoing.x, 1.f / world_outgoing.y,
                                                                     1.f / world_outgoing.z);

               // TODO replace with BRDF refl
               src = src * make_float3(0.8f, 0.5f, 0.5f);

               if (debug) {
                  printf("    bounce ray:\n");
                  printf("    o: %f %f %f\n", origin.x, origin.y, origin.z);
                  printf("    d: %f %f %f\n", direction.x, direction.y, direction.z);
               }

               // todo reflect / bounce according to BRDF

            } 
            else {
               if (debug) {
                  printf("    no hit, becoming inactive\n");
//                     printf("    o: %f %f %f\n", ro.x, ro.y, ro.z);
//                     printf("    d: %f %f %f\n", rd.x, rd.y, rd.z);
               }
               active = false;
            }
         }
         else {
            if (debug) {
               printf("    inactive\n");
            }
            break;
         }

         num_bounces++;
      }
      src *= 255.f;
      device_pointers.device_samples->r[pixel_index] = src.x;
      device_pointers.device_samples->g[pixel_index] = src.y;
      device_pointers.device_samples->b[pixel_index] = src.z;
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
   
   __global__ void generate_camera_rays_centered_kernel(
         float3* const sample_origins,
         float3* const sample_directions,
         float3* const sample_directions_inverse,
         const unsigned int width, const float height, const float tan_fov_half) {
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      bool debug = pixel_index == 131328;

      const float width_f = (float)width;
      const float pixel_x = (float) (pixel_index % width);
      const float pixel_y = (float) (pixel_index / width);

      const float aspect = width_f / height;
      const float pixel_ndc_x = (pixel_x + 0.5f) / width_f;
      const float pixel_ndc_y = (pixel_y + 0.5f) / height;
      
      float3 camera_direction = {
            (2 * pixel_ndc_x - 1) * aspect * tan_fov_half,
            (1 - 2 * pixel_ndc_y) * tan_fov_half,
            // TODO -1 for right-handed
            1
      };

      if (debug) {
         float3 v = camera_direction;
         
         const float length = norm3df(v.x, v.y, v.z);
         const float one_over_length = 1.f / length;
//      const float one_over_length = 1.f / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
         printf("<<<%i, %i>>> unnormalized camera direction (%f, %f, %f) with length %f\n",
                blockIdx.x, threadIdx.x,
                camera_direction.x, camera_direction.y, camera_direction.z, length);
         
         
      }
      
      normalize(camera_direction);

      if (debug) {
         printf("<<<%i, %i>>> normalized camera direction (%f, %f, %f)\n",
                blockIdx.x, threadIdx.x,
                camera_direction.x, camera_direction.y, camera_direction.z);
      }
      float3 world_direction = matrix_apply_vector(camera_to_world_matrix, camera_direction);

      if (debug) {
         printf("<<<%i, %i>>> unnormalized world direction (%f, %f, %f)\n",
                blockIdx.x, threadIdx.x,
                world_direction.x, world_direction.y, world_direction.z);
      }
      normalize(world_direction);

      if (debug) {
         printf("<<<%i, %i>>> normalized world direction (%f, %f, %f)\n",
                blockIdx.x, threadIdx.x,
                world_direction.x, world_direction.y, world_direction.z);
      }
      
      sample_origins[pixel_index] = matrix_apply_point(camera_to_world_matrix, {0, 0, 0});
      sample_directions[pixel_index] = world_direction;
      sample_directions_inverse[pixel_index] = {
            1.f / world_direction.x,
            1.f / world_direction.y,
            1.f / world_direction.z
      };
      
      
      if (debug) {
         const float3 origin = sample_origins[pixel_index];
         const float3 direction = sample_directions[pixel_index];
         const float3 direction_inverse = sample_directions_inverse[pixel_index];
         
         printf("<<<%i, %i>>> generated camera ray o (%f, %f, %f) d (%f %f %f), id (%f %f %f)\n",
                blockIdx.x, threadIdx.x,
                origin.x, origin.y, origin.z,
                direction.x, direction.y, direction.z,
                direction_inverse.x, direction_inverse.y, direction_inverse.z);
      }
      
   }
   
   void PathTracerKernel::Trace() const {
      
      struct device_pointers device_pointers {
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

      const size_t sample_bytes = sizeof(float3) * width * height;
      
      float3* sample_origins;
      float3* sample_directions;
      float3* sample_directions_inverse;
      
      cuda_check_error( cudaMalloc((void**)&sample_origins, sample_bytes) );
      cuda_check_error( cudaMalloc((void**)&sample_directions, sample_bytes) );
      cuda_check_error( cudaMalloc((void**)&sample_directions_inverse, sample_bytes) );

      cudaError_t error = cudaSuccess;
      
      generate_camera_rays_centered_kernel<<<blocksPerGrid, threads_per_block>>>(
            sample_origins, sample_directions, sample_directions_inverse, 
            width, height, tan_fov_half);
      cudaDeviceSynchronize();
      error = cudaGetLastError();

      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to launch generate_camera_rays_centered_kernel (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      // for sample_num = 0 to num_samples
         
         // generate rays
      
         // generate sample_results
      
         // run path tracer => sample_results
      
      // consolidate samples
      
      printf("launching with <<<%i, %i>>>\n", blocksPerGrid, threads_per_block);
      path_trace_kernel<<<blocksPerGrid, threads_per_block>>>(
            sample_origins, 
            sample_directions, 
            sample_directions_inverse,
            width, 
            height, 
            device_pointers);

      cudaDeviceSynchronize();
      error = cudaGetLastError();

      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to launch path_trace_kernel (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }

      cuda_check_error( cudaFree(sample_origins) );
      cuda_check_error( cudaFree(sample_directions) );
      cuda_check_error( cudaFree(sample_directions_inverse) );
   }
}
