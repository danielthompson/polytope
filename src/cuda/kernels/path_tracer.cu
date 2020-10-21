
// Created by daniel on 5/15/20.
//

#include <cmath>
#include <curand_kernel.h>
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

   __constant__ struct device_pointers const_device_pointers;


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
         
         struct device_intersection* intersection,
         const float3 origin,
         const float3 direction
         ) {

      if (intersection->hits) {

         // calculate normal at hit point
         const DeviceMesh mesh_hit = const_device_pointers.device_meshes[intersection->mesh_index];

         const unsigned int v1_index = intersection->face_index + mesh_hit.num_faces;
         const unsigned int v2_index = intersection->face_index + mesh_hit.num_faces * 2;

         const float3 v0 = {mesh_hit.x[intersection->face_index],
                            mesh_hit.y[intersection->face_index],
                            mesh_hit.z[intersection->face_index]};
         const float3 v1 = {mesh_hit.x[v1_index],
                            mesh_hit.y[v1_index],
                            mesh_hit.z[v1_index]};
         const float3 v2 = {mesh_hit.x[v2_index],
                            mesh_hit.y[v2_index],
                            mesh_hit.z[v2_index]};

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
         intersection->hit_point.x = fmaf(n.x, 0.002f, intersection->hit_point.x);
         intersection->hit_point.y = fmaf(n.y, 0.002f, intersection->hit_point.y);
         intersection->hit_point.z = fmaf(n.z, 0.002f, intersection->hit_point.z);

         const float edge0dot = fabs(dot(e0, e1));
         const float edge1dot = fabs(dot(e1, e2));
         const float edge2dot = fabs(dot(e2, e0));

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
   
   __device__ void bvh_intersect(
         struct device_intersection* intersection,
         const float3 origin,
         const float3 direction,
         const float3 direction_inverse,
         const unsigned int bounce_num
         //struct device_pointers device_pointers
         ) {
      
      // TODO - figure out a bound on how big this could/should be
      int future_node_stack[1024];
      
      /**
       * first free index in future_node_stack
       */
      int next_future_node_index = 0;
      future_node_stack[next_future_node_index] = 0;

      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
//      bool debug = pixel_index == 131328;
      bool debug = false;
      
      poly::device_bvh_node node;
      int current_node_index = 0;
         
      do {
         node = const_device_pointers.device_bvh_node[current_node_index];
         cuda_debug_printf(debug, "Looking at node %i\n", current_node_index);
         const bool is_leaf = node.is_leaf();
         cuda_debug_printf(debug, "  Intersecting node %i with ray o (%f, %f, %f) d (%f %f %f), id (%f %f %f)\n", 
                current_node_index,
                origin.x, origin.y, origin.z,
                direction.x, direction.y, direction.z,
                direction_inverse.x, direction_inverse.y, direction_inverse.z);
         
         float aabb_t = INFINITY;
         
         if (aabb_hits(node.aabb, origin, direction_inverse, &aabb_t) && aabb_t < intersection->t) {
            cuda_debug_printf(debug, "  Ray hit node %i aabb\n", current_node_index);
            if (is_leaf) {
               cuda_debug_printf(debug, "  Node %i is leaf, intersecting faces\n", current_node_index);
               for (int i = 0; i < node.num_faces; i++) {

                  device_index_pair indices = const_device_pointers.device_index_pair[node.offset + i];
                  const unsigned int mesh_index = indices.mesh_index;
                  const unsigned int face_index = indices.face_index;
                  cuda_debug_printf(debug, "Testing face_index %i\n", indices.face_index);
                  const DeviceMesh mesh = const_device_pointers.device_meshes[mesh_index];
                  
                  const unsigned int v1_index = face_index + (mesh.num_faces);
                  const unsigned int v2_index = face_index + (mesh.num_faces * 2);

                  const float3 v0 = make_float3(__ldg(&mesh.x[face_index]), 
                                                __ldg(&mesh.y[face_index]), 
                                                __ldg(&mesh.z[face_index]));
                  const float3 v1 = make_float3(__ldg(&mesh.x[v1_index]),
                                                __ldg(&mesh.y[v1_index]),
                                                __ldg(&mesh.z[v1_index]));
                  const float3 v2 = make_float3(__ldg(&mesh.x[v2_index]),
                                                __ldg(&mesh.y[v2_index]),
                                                __ldg(&mesh.z[v2_index]));

                  const float3 e0 = v1 - v0;
                  const float3 e1 = v2 - v1;
                  float3 pn = cross(e0, e1);

                  pn *= __frsqrt_rn(dot(pn, pn));
                  float ft = INFINITY;
                  
                  {
                     const float divisor = dot(pn, direction);
                     if (divisor == 0.0f) {
                        cuda_debug_printf(debug, "Ray is parallel to face %i, bailing\n", face_index);
                        continue;
                     }

                     ft = (dot(pn, v0 - origin)) / divisor;
                  }

                  if (ft <= 0 || ft > intersection->t) {
                     cuda_debug_printf(debug, "Bailing on face %i due to previous t %f vs this t %f \n", face_index,
                               intersection->t, ft);
                     continue;
                  }

                  const float3 hp = origin + direction * ft;

                  {
                     float3 p = hp - v0;
                     float3 cross_ep = cross(e0, p);
                     float normal = dot(cross_ep, pn);
                     if (normal <= 0) {
                        cuda_debug_printf(debug, "Bailing on face %i due to first normal test %f\n", face_index, normal);
                        continue;
                     }

                     p = hp - v1;
                     cross_ep = cross(e1, p);
                     normal = dot(cross_ep, pn);
                     if (normal <= 0) {
                        
                        cuda_debug_printf(debug, "Bailing on face %i due to second normal test %f\n", face_index, normal);
                        continue;
                     }

                     const float3 e2 = v0 - v2;
                     p = hp - v2;
                     cross_ep = cross(e2, p);
                     normal = dot(cross_ep, pn);
                     if (normal <= 0) {
                        cuda_debug_printf(debug, "Bailing on face %i due to third normal test %f\n", face_index, normal);
                        continue;
                     }
                  }

                  cuda_debug_printf(debug, "Hit face face_index %i with t %f\n", face_index, ft);
                  // temp
                  intersection->hits = true;
                  intersection->t = ft;
                  intersection->face_index = face_index;
                  intersection->mesh_index = mesh_index;
               }
               

               // next node should be from the stack, if any
               if (next_future_node_index == 0) {
                  cuda_debug_printf(debug, "  No more nodes on stack, ending traversal\n");
                  break;
               }
               
               cuda_debug_printf(debug, "  Picking up next node from stack\n");
               current_node_index = future_node_stack[--next_future_node_index];
            } else {

               cuda_debug_printf(debug, "  Node %i is interior, continuing traversal\n", current_node_index);
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

            cuda_debug_printf(debug, "  Ray missed node %i aabb\n", current_node_index);
            // next node should be from the stack
            if (next_future_node_index == 0) {
               cuda_debug_printf(debug, "  No more nodes on stack, ending traversal with hit %i and t %f\n", (int)intersection->hits, intersection->t);
               break;
            }
            
            cuda_debug_printf(debug, "  Picking up next node from stack\n");
            current_node_index = future_node_stack[--next_future_node_index];
         }
      } while (current_node_index >= 0);
      
      consolidate_sample_intersections(
            intersection,
            origin,
            direction
            );
   }
   
   __global__ void path_trace_kernel(
         float3* const origins,
         float3* const directions,
         float3* const directions_inverse,
         curandState_t* const rng_states 
               ) {
      // loop over pixels
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;

      bool debug = false;
//      bool debug = pixel_index == 131328;

      float3 src = { 1.0f, 1.0f, 1.0f};
      
      bool active = true;
      
      device_intersection intersection;
      
      float3 o = origins[pixel_index];
      float3 d = directions[pixel_index];
      float3 di = directions_inverse[pixel_index];
      
      unsigned int num_bounces = 0;
      while (true) {
         
         cuda_debug_printf(debug, "Bounce %i\n", num_bounces);
         
         if (num_bounces > 5) {
            break;
         }
         
         // process each sample
         if (active) {

            // reset all intersections
            intersection.face_index = 0;
            intersection.mesh_index = 0;
            intersection.hit_point = {0.f, 0.f, 0.f };
            intersection.t = INFINITY;
            intersection.hits = false;
            bvh_intersect(
                  &intersection,
                  o,
                  d,
                  di,
                  num_bounces
                  //device_pointers
                  );
            
            if (intersection.hits) {
               cuda_debug_printf(debug, "  Hit mesh %i face %i with ray o: (%f %f %f) d: (%f %f %f)\n", 
                                 intersection.mesh_index, intersection.face_index, o.x, o.y, o.z, d.x, d.y, d.z);
               
               // todo hit light

               // TODO put this into a lambert_brdf function
//                const poly::Vector local_incoming = intersection.WorldToLocal(current_ray.Direction);
               float3 local_incoming = normalize(
                     make_float3(dot(d, intersection.tangent1), 
                                 dot(d, intersection.normal),
                                 dot(d, intersection.tangent2)));

               // mirror
//               const float3 local_outgoing = make_float3(local_incoming.x, -local_incoming.y, local_incoming.z);

               // lambert
               float u0 = curand_uniform(&rng_states[pixel_index]);
               float u1 = curand_uniform(&rng_states[pixel_index]);
               float3 local_outgoing = cosine_sample_hemisphere(u0, u1);
               float3 foo = cosine_sample_hemisphere(0.5f, 0.5f);

               float3 world_outgoing = normalize(make_float3(
                     intersection.tangent1.x * local_outgoing.x + intersection.normal.x * local_outgoing.y +
                     intersection.tangent2.x * local_outgoing.z,
                     intersection.tangent1.y * local_outgoing.x + intersection.normal.y * local_outgoing.y +
                     intersection.tangent2.y * local_outgoing.z,
                     intersection.tangent1.z * local_outgoing.x + intersection.normal.z * local_outgoing.y +
                     intersection.tangent2.z * local_outgoing.z
               ));

               // TOOD offset
               o = intersection.hit_point;
               d = world_outgoing;
               di = make_float3(1.f / world_outgoing.x, 1.f / world_outgoing.y,
                                                                     1.f / world_outgoing.z);

               // TODO replace with BRDF refl
               src = src * make_float3(0.8f, 0.5f, 0.5f);

               cuda_debug_printf(debug, "  bounce ray o: (%f %f %f) d: (%f %f %f)\n", o.x, o.y, o.z, d.x, d.y, d.z);

               // todo reflect / bounce according to BRDF
            } 
            else {
               cuda_debug_printf(debug, " no hit, becoming inactive\n");
               active = false;
            }
         }
         else {
            cuda_debug_printf(debug, " inactive\n");
            break;
         }

         num_bounces++;
      }
      
      src *= 255.f;
      const_device_pointers.device_samples->r[pixel_index] += src.x;
      const_device_pointers.device_samples->g[pixel_index] += src.y;
      const_device_pointers.device_samples->b[pixel_index] += src.z;
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
      bool debug = false;
      //bool debug = pixel_index == 131328;

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

      normalize(camera_direction);
      cuda_debug_printf(debug, "Normalized camera direction (%f, %f, %f)\n",
                camera_direction.x, camera_direction.y, camera_direction.z);
      float3 world_direction = matrix_apply_vector(camera_to_world_matrix, camera_direction);
      cuda_debug_printf(debug, "Unnormalized world direction (%f, %f, %f)\n",
                world_direction.x, world_direction.y, world_direction.z);
      
      normalize(world_direction);
      cuda_debug_printf(debug, "Normalized world direction (%f, %f, %f)\n",
                world_direction.x, world_direction.y, world_direction.z);
      
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

         cuda_debug_printf(debug, "Generated camera ray o (%f, %f, %f) d (%f %f %f), id (%f %f %f)\n",
                origin.x, origin.y, origin.z,
                direction.x, direction.y, direction.z,
                direction_inverse.x, direction_inverse.y, direction_inverse.z);
      }
   }
   
   __global__ void generate_camera_rays_random_kernel(
         float3* const sample_origins,
         float3* const sample_directions,
         float3* const sample_directions_inverse,
         curandState_t* const rng_states,
         const unsigned int width, const float height, const float tan_fov_half) {
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      
      const float width_f = (float)width;
      const float pixel_x = (float) (pixel_index % width);
      const float pixel_y = (float) (pixel_index / width);

      const float x_offset = curand_uniform(&rng_states[pixel_index]);
      const float y_offset = curand_uniform(&rng_states[pixel_index]);

      const float aspect = width_f / height;
      const float pixel_ndc_x = (pixel_x + x_offset) / width_f;
      const float pixel_ndc_y = (pixel_y + y_offset) / height;

      float3 camera_direction = {
            (2 * pixel_ndc_x - 1) * aspect * tan_fov_half,
            (1 - 2 * pixel_ndc_y) * tan_fov_half,
            // TODO -1 for right-handed
            1
      };

      normalize(camera_direction);
      float3 world_direction = matrix_apply_vector(camera_to_world_matrix, camera_direction);
      normalize(world_direction);
      sample_origins[pixel_index] = matrix_apply_point(camera_to_world_matrix, {0, 0, 0});
      sample_directions[pixel_index] = world_direction;
      sample_directions_inverse[pixel_index] = {
            1.f / world_direction.x,
            1.f / world_direction.y,
            1.f / world_direction.z
      };
   }
   
   __global__ void init_curand_states_kernel(curandState_t* states, const unsigned int num_elements) {
      const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
      if (index < num_elements)
         curand_init(index, 0, 0, &states[index]);
      
   }
   
   __global__ void reduce_samples_kernel(const float inv_num_samples, const unsigned int num_elements) {
      const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
      if (index < num_elements) {
         const_device_pointers.device_samples->r[index] *= inv_num_samples;
         const_device_pointers.device_samples->g[index] *= inv_num_samples;
         const_device_pointers.device_samples->b[index] *= inv_num_samples;
      }
   }
   
   void PathTracerKernel::Trace(unsigned int num_samples) const {
      
      struct device_pointers device_pointers {
         memory_manager->meshes,
         memory_manager->num_meshes,
         memory_manager->device_samples,
         memory_manager->device_bvh,
         memory_manager->num_bvh_nodes,
         memory_manager->index_pair,
         memory_manager->num_indices
      };

      cudaError_t error = cudaSuccess;
      
      const float tan_fov_half = std::tan(memory_manager->camera_fov * poly::PIOver360);
      
      cuda_check_error( cudaMemcpyToSymbol(camera_to_world_matrix, memory_manager->camera_to_world_matrix, sizeof(float) * 16));
      cuda_check_error( cudaMemcpyToSymbol(const_device_pointers, &device_pointers, sizeof(struct device_pointers)) );
      
      const unsigned int blocksPerGrid = (memory_manager->num_pixels + threads_per_block - 1) / threads_per_block;

      curandState_t* device_states;
      cuda_check_error( cudaMalloc((void**)&device_states, sizeof(curandState_t) * memory_manager->num_pixels));
      {
         const unsigned int curand_init_tpb = 256;
         const unsigned int curand_init_bgp = (memory_manager->num_pixels + curand_init_tpb - 1) / curand_init_tpb;

         //printf("launching init_curand_states_kernel<<<%i, %i>>>\n", curand_init_bgp, curand_init_tpb);
         init_curand_states_kernel<<<curand_init_bgp, curand_init_tpb>>>(device_states, memory_manager->num_pixels);
         cudaDeviceSynchronize();
         error = cudaGetLastError();
   
         if (error != cudaSuccess)
         {
            fprintf(stderr, "Failed to launch init_curand_states_kernel (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
         }
      }
      
      const unsigned int width = memory_manager->width;
      const unsigned int height = memory_manager->height;

      const size_t sample_bytes = sizeof(float3) * width * height;
      
      float3* sample_origins;
      float3* sample_directions;
      float3* sample_directions_inverse;
      
      cuda_check_error( cudaMalloc((void**)&sample_origins, sample_bytes) );
      cuda_check_error( cudaMalloc((void**)&sample_directions, sample_bytes) );
      cuda_check_error( cudaMalloc((void**)&sample_directions_inverse, sample_bytes) );

      for (int i = 0; i < num_samples; i++) {

         // generate rays
         //printf("launching generate_camera_rays_centered_kernel<<<%i, %i>>>\n", blocksPerGrid, threads_per_block);
         generate_camera_rays_random_kernel<<<blocksPerGrid, threads_per_block>>>(
               sample_origins, sample_directions, sample_directions_inverse, device_states,
               width, (float) height, tan_fov_half);
         cudaDeviceSynchronize();
         error = cudaGetLastError();

         if (error != cudaSuccess) {
            fprintf(stderr, "Failed to launch generate_camera_rays_centered_kernel (error code %s)!\n",
                    cudaGetErrorString(error));
            exit(EXIT_FAILURE);
         }


         // run path tracer => sample_results

         //printf("launching path_trace_kernel<<<%i, %i>>>\n", blocksPerGrid, threads_per_block);
         path_trace_kernel<<<blocksPerGrid, threads_per_block>>>(
               sample_origins,
               sample_directions,
               sample_directions_inverse,
               device_states
         );

         cudaDeviceSynchronize();
         error = cudaGetLastError();

         if (error != cudaSuccess) {
            fprintf(stderr, "Failed to launch path_trace_kernel (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
         }
      }

      // consolidate samples

      const float num_samples_inv = 1.f / (float)num_samples;
      //printf("launching reduce_samples_kernel<<<%i, %i>>>\n", blocksPerGrid, threads_per_block);
      reduce_samples_kernel<<<blocksPerGrid, threads_per_block>>>(num_samples_inv, memory_manager->num_pixels);

      cudaDeviceSynchronize();
      error = cudaGetLastError();

      if (error != cudaSuccess) {
         fprintf(stderr, "Failed to launch reduce_samples_kernel (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      cuda_check_error( cudaFree(sample_origins) );
      cuda_check_error( cudaFree(sample_directions) );
      cuda_check_error( cudaFree(sample_directions_inverse) );
      cuda_check_error( cudaFree(device_states) );
   }
}
