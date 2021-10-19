
// Created by daniel on 5/15/20.
//

#include <cmath>
#include <curand_kernel.h>
#include "path_tracer.cuh"
#include "common_device_functions.cuh"
#include "../check_error.h"
#include <cassert>

namespace poly {

   
//#define DEBUG_PIXEL_INDEX 164160 /* 320, 320 */
#define DEBUG_PIXEL_X 512
#define DEBUG_PIXEL_Y 175

#define DEBUG_PIXEL_INDEX DEBUG_PIXEL_Y * 1024 + DEBUG_PIXEL_X  /* 192, 192 */
#define DEBUG_PRINT 0
   
   constexpr unsigned int threads_per_block = 32;
   
   constexpr unsigned int num_constant_nodes = 1536;
   __constant__ poly::device_bvh_node const_bvh_nodes[num_constant_nodes];


   __constant__ float camera_to_world_matrix[16];
   
   struct device_pointers {
      struct poly::device_mesh* device_meshes;
      unsigned int num_meshes;

      struct poly::device_mesh_geometry* device_mesh_geometry;
      
      struct poly::Samples* device_samples;
      
      poly::device_bvh_node* device_bvh_node;
      unsigned int num_bvh_nodes;
      
      poly::device_index_pair* device_index_pair;
      unsigned int num_index_pairs;
   };

   __constant__ struct poly::device_pointers const_device_pointers;
   __constant__ poly::device_bvh_node root_node; 

   struct device_intersection {
      float t;
      unsigned int mesh_index;
      unsigned int face_index;
      float3 normal;
      float3 hit_point;
      float3 tangent1;
      float3 tangent2;
      bool hits;
      float u, v, w, new_t;
   };
   
   __device__ float dim(const float3 v, const int d) {
      switch (d) {
         case 0:
            return v.x;
         case 1:
            return v.y;
         default:
            return v.z;
      }
   }
   
   __device__ float3 mirror_sample(float3 local_incoming) {
      return make_float3(local_incoming.x, -local_incoming.y, local_incoming.z);
   }
   
   __device__ float3 lambert_sample(float3 local_incoming, curandState_t* const rng) {
      float u0 = curand_uniform(rng);
      float u1 = curand_uniform(rng);
      return cosine_sample_hemisphere(u0, u1);
   }
   
   __device__ bool aabb_hits(const float* aabb, const float3 origin, const float3 inverse_direction, float* const mint) {

      // TODO take ray's current t as a param and use for maxBoundFarT
      float maxBoundFarT = poly::Infinity;
      float minBoundNearT = 0;
      
      // TODO
      const float gammaMultiplier = 1 + 2 * poly::Gamma3;

      // X
      float tNear = (aabb[0] - origin.x) * inverse_direction.x;
      float tFar = (aabb[3] - origin.x) * inverse_direction.x;

      float swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      // TODO
      tFar *= gammaMultiplier;

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
      tFar *= gammaMultiplier;

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
      tFar *= gammaMultiplier;

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
   
   bool path_tracer::unit_test_hit_ray_against_bounding_box(const poly::ray &ray, const float* const device_aabb) {
      
      int* device_result;
      cuda_check_error( cudaMalloc((void**)&device_result, sizeof(int)) );
      
      float3 o = make_float3(ray.origin.x, ray.origin.y, ray.origin.z);
      float3 id = {1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z};

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
         
         struct poly::device_intersection* intersection,
         const float3 origin,
         const float3 direction
         ) {

      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      bool debug = false;
      if (DEBUG_PRINT) {
         debug = (pixel_index == DEBUG_PIXEL_INDEX);
      }
      
      if (intersection->hits) {

         // calculate normal at hit point
         // TOOD precalculate face normals
         const device_mesh mesh_hit = const_device_pointers.device_meshes[intersection->mesh_index];
         const device_mesh_geometry geometry = const_device_pointers.device_mesh_geometry[mesh_hit.device_mesh_geometry_offset];
         
         const unsigned int v1_index = intersection->face_index + geometry.num_faces;
         const unsigned int v2_index = intersection->face_index + geometry.num_faces * 2;

         float3 v0 = {geometry.x[intersection->face_index],
                      geometry.y[intersection->face_index],
                      geometry.z[intersection->face_index]};
         float3 v1 = {geometry.x[v1_index],
                      geometry.y[v1_index],
                      geometry.z[v1_index]};
         float3 v2 = {geometry.x[v2_index],
                      geometry.y[v2_index],
                      geometry.z[v2_index]};
         
         v0 = matrix_apply_point(mesh_hit.obj_to_world, v0);
         v1 = matrix_apply_point(mesh_hit.obj_to_world, v1);
         v2 = matrix_apply_point(mesh_hit.obj_to_world, v2);
         
         cuda_debug_printf(debug, "  Face %i v0 %i: (%f %f %f), v1 %i: (%f %f %f), v2 %i: (%f %f %f)\n", 
                           intersection->face_index, intersection->face_index, v0.x, v0.y, v0.z, v1_index, v1.x, v1.y, v1.z, v2_index, v2.x, v2.y, v2.z);
         
         const float3 e0 = v1 - v0;
         const float3 e1 = v2 - v1;
         const float3 e2 = v0 - v2;
         
         float3 n;

         if (geometry.has_vertex_normals) {
            cuda_debug_printf(debug, "  Vertex normals\n");
            cuda_debug_printf(debug, "    u %f v %f w %f\n", intersection->u, intersection->v, intersection->w);
            float3 v0n = {geometry.nx[intersection->face_index],
                          geometry.ny[intersection->face_index],
                          geometry.nz[intersection->face_index]};
            cuda_debug_printf(debug, "    v0n: (%f, %f, %f)\n", v0n.x, v0n.y, v0n.z);
            float3 v1n = {geometry.nx[v1_index],
                          geometry.ny[v1_index],
                          geometry.nz[v1_index]};
            cuda_debug_printf(debug, "    v1n: (%f, %f, %f)\n", v1n.x, v1n.y, v1n.z);
            float3 v2n = {geometry.nx[v2_index],
                          geometry.ny[v2_index],
                          geometry.nz[v2_index]};
            cuda_debug_printf(debug, "    v2n: (%f, %f, %f)\n", v2n.x, v2n.y, v2n.z);

            // use inverse matrix, i.e. world_to_object
            v0n = matrix_apply_normal(mesh_hit.world_to_object, v0n);
            v1n = matrix_apply_normal(mesh_hit.world_to_object, v1n);
            v2n = matrix_apply_normal(mesh_hit.world_to_object, v2n);
            
            n = v0n * intersection->u + v1n * intersection->v + v2n * intersection->w;
            cuda_debug_printf(debug, "    Interpolated normal: (%f, %f, %f)\n", n.x, n.y, n.z);
         }
         else {
            n = cross(e0, e1);
            cuda_debug_printf(debug, "  No vertex normals, using face normal (%f, %f, %f)\n", n.x, n.y, n.z);
         }
         // flip normal if needed
         const float ray_dot_normal = dot(direction, n);
         const float flip_factor = ray_dot_normal > 0 ? -1 : 1;
         n *= flip_factor;
         normalize(n);
         intersection->normal = n;
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
         struct poly::device_intersection* intersection,
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
      bool debug = false;
      if (DEBUG_PRINT) {
         debug = (pixel_index == DEBUG_PIXEL_INDEX);
      }

      poly::device_bvh_node node;
      int current_node_index = 0;

      // ray precalculation
      float dx_abs = fabs(direction.x);
      float dy_abs = fabs(direction.y);
      float dz_abs = fabs(direction.z);

      int kz = 0;
      if (dy_abs > dx_abs && dy_abs > dz_abs)
         kz = 1;
      else if (dz_abs > dx_abs && dz_abs > dy_abs)
         kz = 2;

      int kx = kz + 1;
      if (kx == 3)
         kx = 0;
      int ky = kx + 1;
      if (ky == 3)
         ky = 0;

      // swap kx and ky dimension to preserve winding direction of triangles
      if (dim(direction, kz) < 0.0f) {
         // swap(kx, ky)
         int temp = kx;
         kx = ky;
         ky = temp;
      }

      // calculate shear constants
      float Sx = dim(direction, kx) / dim(direction, kz);
      float Sy = dim(direction, ky) / dim(direction, kz);
      float Sz = 1.0f / dim(direction, kz);
      
      do {
         if (current_node_index < num_constant_nodes) {
            node = const_bvh_nodes[current_node_index];
         }
         else {
            node = const_device_pointers.device_bvh_node[current_node_index];
         }
         //node = const_device_pointers.device_bvh_node[current_node_index];
         cuda_debug_printf(debug, "Looking at node %i\n", current_node_index);
         const bool is_leaf = node.is_leaf();
         cuda_debug_printf(debug, "  Intersecting node %i with ray o (%f, %f, %f) d (%f %f %f), id (%f %f %f)\n", 
                current_node_index,
                origin.x, origin.y, origin.z,
                direction.x, direction.y, direction.z,
                direction_inverse.x, direction_inverse.y, direction_inverse.z);
         
         float aabb_t = INFINITY;
         
         if (aabb_hits(node.bb, origin, direction_inverse, &aabb_t) && aabb_t < intersection->t) {
            cuda_debug_printf(debug, "  Ray hit node %i aabb\n", current_node_index);
            if (is_leaf) {
               cuda_debug_printf(debug, "  Node %i is leaf, intersecting faces\n", current_node_index);
               for (int i = 0; i < node.num_faces; i++) {

                  poly::device_index_pair indices = const_device_pointers.device_index_pair[node.offset + i];
                  const unsigned int mesh_index = indices.mesh_index;
                  const unsigned int face_index = indices.face_index;
                  cuda_debug_printf(debug, "    Testing face_index %i\n", indices.face_index);
                  const poly::device_mesh mesh = const_device_pointers.device_meshes[mesh_index];
                  cuda_debug_printf(debug, "      Loaded mesh_index %i\n", mesh_index);
                  cuda_debug_printf(debug, "      obj_to_world: \t%f\t%f\t%f\t%f\n", mesh.obj_to_world[0], mesh.obj_to_world[1], mesh.obj_to_world[2], mesh.obj_to_world[3]);
                  cuda_debug_printf(debug, "                    \t%f\t%f\t%f\t%f\n", mesh.obj_to_world[4], mesh.obj_to_world[5], mesh.obj_to_world[6], mesh.obj_to_world[7]);
                  cuda_debug_printf(debug, "                    \t%f\t%f\t%f\t%f\n", mesh.obj_to_world[8], mesh.obj_to_world[9], mesh.obj_to_world[10], mesh.obj_to_world[11]);
                  cuda_debug_printf(debug, "                    \t%f\t%f\t%f\t%f\n", mesh.obj_to_world[12], mesh.obj_to_world[13], mesh.obj_to_world[14], mesh.obj_to_world[15]);

                  cuda_debug_printf(debug, "      world_to_obj: \t%f\t%f\t%f\t%f\n", mesh.world_to_object[0], mesh.world_to_object[1], mesh.world_to_object[2], mesh.world_to_object[3]);
                  cuda_debug_printf(debug, "                    \t%f\t%f\t%f\t%f\n", mesh.world_to_object[4], mesh.world_to_object[5], mesh.world_to_object[6], mesh.world_to_object[7]);
                  cuda_debug_printf(debug, "                    \t%f\t%f\t%f\t%f\n", mesh.world_to_object[8], mesh.world_to_object[9], mesh.world_to_object[10], mesh.world_to_object[11]);
                  cuda_debug_printf(debug, "                    \t%f\t%f\t%f\t%f\n", mesh.world_to_object[12], mesh.world_to_object[13], mesh.world_to_object[14], mesh.world_to_object[15]);


                  const poly::device_mesh_geometry geometry = const_device_pointers.device_mesh_geometry[mesh.device_mesh_geometry_offset];
                  cuda_debug_printf(debug, "      Loaded geometry %zu\n", mesh.device_mesh_geometry_offset);
                  
                  const unsigned int v1_index = face_index + (geometry.num_faces);
                  const unsigned int v2_index = face_index + (geometry.num_faces * 2);

                  float3 v0 = make_float3(__ldg(&geometry.x[face_index]), 
                                                __ldg(&geometry.y[face_index]), 
                                                __ldg(&geometry.z[face_index]));
                  cuda_debug_printf(debug, "      Loaded v0 %f %f %f\n", v0.x, v0.y, v0.z);
                  float3 v0_world = matrix_apply_point(mesh.obj_to_world, v0);
                  
                  float3 v1 = make_float3(__ldg(&geometry.x[v1_index]),
                                                __ldg(&geometry.y[v1_index]),
                                                __ldg(&geometry.z[v1_index]));
                  cuda_debug_printf(debug, "      Loaded v1 %f %f %f\n", v1.x, v1.y, v1.z);
                  float3 v1_world = matrix_apply_point(mesh.obj_to_world, v1);
                  
                  float3 v2 = make_float3(__ldg(&geometry.x[v2_index]),
                                                __ldg(&geometry.y[v2_index]),
                                                __ldg(&geometry.z[v2_index]));
                  cuda_debug_printf(debug, "      Loaded v2 %f %f %f\n", v2.x, v2.y, v2.z);
                  float3 v2_world = matrix_apply_point(mesh.obj_to_world, v2);
                  
                  cuda_debug_printf(debug, "      World space: %f %f %f\n", v0_world.x, v0_world.y, v0_world.z);
                  cuda_debug_printf(debug, "      World space: %f %f %f\n", v1_world.x, v1_world.y, v1_world.z);
                  cuda_debug_printf(debug, "      World space: %f %f %f\n", v2_world.x, v2_world.y, v2_world.z);

                  // wald watertight intersection
                  // http://jcgt.org/published/0002/01/05/paper.pdf

                  // calculate dimension where the ray direction is maximal 
                  //int kz = max_dim(abs(dir));
                  
                  // calculate vertices relative to ray origin
                  const float3 A = v0_world - origin;
                  const float3 B = v1_world - origin;
                  const float3 C = v2_world - origin;
                  
                  // perform shear and scale of vertices
                  const float Ax = dim(A, kx) - Sx * dim(A, kz);
                  const float Ay = dim(A, ky) - Sy * dim(A, kz);
                  const float Bx = dim(B, kx) - Sx * dim(B, kz);
                  const float By = dim(B, ky) - Sy * dim(B, kz);
                  const float Cx = dim(C, kx) - Sx * dim(C, kz);
                  const float Cy = dim(C, ky) - Sy * dim(C, kz);
                  
                  // calculate scaled barycentric coordinates
                  float U = Cx * By - Cy * Bx;
                  float V = Ax * Cy - Ay * Cx;
                  float W = Bx * Ay - By * Ax;
                  
                  // fallback to test against edges using double precision
                  if (U == 0.0f || V == 0.0f || W == 0.0f) {
                     double CxBy = (double)Cx*(double)By;
                     double CyBx = (double)Cy*(double)Bx;
                     U = (float)(CxBy - CyBx);
                     
                     double AxCy = (double)Ax*(double)Cy;
                     double AyCx = (double)Ay*(double)Cx;
                     V = (float)(AxCy - AyCx);
                     
                     double BxAy = (double)Bx*(double)Ay;
                     double ByAx = (double)By*(double)Ax;
                     W = (float)(BxAy - ByAx);
                  }
                  
                  // Perform edge tests. Moving this test before and at the end of the previous
                  // conditional gives higher performance.
                  // backface culling:
//                  if (U < 0.0f || V < 0.0f || W < 0.0f)
//                     continue;
                  // no backface culling:
                  if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f))
                     continue;
                  
                  // calculate determinant
                  float det = U + V + W;
                  if (det == 0.0f)
                     continue;
                  
                  // calculate scaled z-coordinates of vertices and use them to calculate the hit distance
                  const float Az = Sz * dim(A, kz);
                  const float Bz = Sz * dim(B, kz);
                  const float Cz = Sz * dim(C, kz);
                  const float T = U * Az + V * Bz + W * Cz;
                  
                  // backface culling
//                  if (T < 0.0f || T > intersection->t * det)
//                     continue;

                  // no backface culling
                  if (det < 0.f && (T >= 0 || T < intersection->t * det)) {
                     continue;
                  }
                  if (det > 0.f && (T <= 0 || T > intersection->t * det)) {
                     continue;
                  }
                  
                  // normalize
                  const float rcpDet = 1.0f / det;
                  float u = U * rcpDet;
                  float v = V * rcpDet;
                  float w = W * rcpDet;
                  intersection->t = T * rcpDet;
                  intersection->hits = true;
                  intersection->face_index = face_index;
                  intersection->mesh_index = mesh_index;
                  intersection->hit_point = v0_world * u + v1_world * v + v2_world * w;
                  intersection->u = u;
                  intersection->v = v;
                  intersection->w = w;
                  cuda_debug_printf(debug, "      Hit with t %f\n", intersection->t);
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
      if (DEBUG_PRINT) {
         debug = (pixel_index == DEBUG_PIXEL_INDEX);
      }

      float3 src = { 1.0f, 1.0f, 1.0f};
      
      bool active = true;

      poly::device_intersection intersection {};
      
      float3 o = origins[pixel_index];
      float3 d = directions[pixel_index];
      float3 di = directions_inverse[pixel_index];
      
      unsigned int num_bounces = 0;
      while (true) {
         
         cuda_debug_printf(debug, "Bounce %i\n", num_bounces);
         
         // TODO also break if all samples are inactive
         if (num_bounces > 20) {
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
               cuda_debug_printf(debug, "Hit mesh %i face %i with t %f ray o: (%f %f %f) d: (%f %f %f)\n", 
                                 intersection.mesh_index, intersection.face_index, intersection.t, o.x, o.y, o.z, d.x, d.y, d.z);
               
               float3 local_incoming = normalize(
                     make_float3(dot(d, intersection.tangent1), 
                                 dot(d, intersection.normal),
                                 dot(d, intersection.tangent2)));

               const poly::device_mesh mesh_hit = const_device_pointers.device_meshes[intersection.mesh_index];

               float3 local_outgoing;

               switch (mesh_hit.brdf_type) {
                  case BRDF_TYPE::Lambert: {
                     local_outgoing = lambert_sample(local_incoming, &(rng_states[pixel_index]));
                     cuda_debug_printf(debug, "  Lambert reflected (local) dir: (%f %f %f)\n",
                                       local_outgoing.x, local_outgoing.y, local_outgoing.z);

                     src = src * make_float3(mesh_hit.brdf_params[0],mesh_hit.brdf_params[1],mesh_hit.brdf_params[2]);
                     break;
                  }
                  case BRDF_TYPE::Mirror: {
                     local_outgoing = mirror_sample(local_incoming);
                     src = src * make_float3(mesh_hit.brdf_params[0],mesh_hit.brdf_params[1],mesh_hit.brdf_params[2]);
                     break;
                  }
                  default:
                     // light
                     src = src * make_float3(mesh_hit.brdf_params[7],mesh_hit.brdf_params[8],mesh_hit.brdf_params[9]); 
                     active = false;
                     break;
               }

               if (!active) {
                  break;
               }
               
               cuda_debug_printf(debug, "  Intersection t1 (%f %f %f) t2 (%f %f %f)\n",
                                 intersection.tangent1.x, intersection.tangent1.y, intersection.tangent1.z,
                                 intersection.tangent2.x, intersection.tangent2.y, intersection.tangent2.z);
               
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
               //src = src * make_float3(0.8f, 0.5f, 0.5f);

               cuda_debug_printf(debug, "  bounce ray o: (%f %f %f) d: (%f %f %f)\n", o.x, o.y, o.z, d.x, d.y, d.z);

               // todo reflect / bounce according to BRDF
            } 
            else {
               cuda_debug_printf(debug, " no hit, becoming inactive\n");
               src = { 0.0f, 0.0f, 0.0f };
               active = false;
               break;
            }
         }
         else {
            cuda_debug_printf(debug, " inactive\n");
            break;
         }

         num_bounces++;
      }
      
      //src *= 255.f;
      const_device_pointers.device_samples->r[pixel_index] += src.x;
      const_device_pointers.device_samples->g[pixel_index] += src.y;
      const_device_pointers.device_samples->b[pixel_index] += src.z;
   }
   
   __global__ void box_filter_kernel(float3* src, struct Samples* device_samples, const unsigned int num_samples) {
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      float3 sample_sum = make_float3(0, 0, 0);

      for (unsigned int sample_index = 0; sample_index < num_samples; sample_index++) {

         //float3 sample = src[sample_index];
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
   
   __global__ void sample_stratified_1d(
         curandState_t* const rng_states,
         float* const samples, 
         int num_samples) {
      const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
      if (index >= num_samples)
         return;
      
      const float stratum_width = 1.f / (float)num_samples;
      const float uniform_sample = curand_uniform(&rng_states[index]);
      const float scaled_uniform_sample = uniform_sample * stratum_width;
      const float stratum_offset = stratum_width * (float)index;
      const float stratified_sample = stratum_offset + scaled_uniform_sample;
      samples[index] = stratified_sample;
   }

   __global__ void sample_stratified_2d(
         curandState_t* const rng_states,
         float* const samples,
         int num_samples_x,
         int num_samples_y) {
      const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
      if (index >= num_samples_x * num_samples_y)
         return;

      const float stratum_width_x = 1.f / (float)num_samples_x;
      const float uniform_sample_x = curand_uniform(&rng_states[index]);
      const float scaled_uniform_sample_x = uniform_sample_x * stratum_width_x;
      const float stratum_offset_x = stratum_width_x * (float)index;
      const float stratified_sample_x = stratum_offset_x + scaled_uniform_sample_x;

      const float stratum_width_y = 1.f / (float)num_samples_y;
      const float uniform_sample_y = curand_uniform(&rng_states[index]);
      const float scaled_uniform_sample_y = uniform_sample_y * stratum_width_y;
      const float stratum_offset_y = stratum_width_y * (float)index;
      const float stratified_sample_y = stratum_offset_y + scaled_uniform_sample_y;
      

      samples[2 * index] = stratified_sample_x;
      samples[2 * index + 1] = stratified_sample_y;
   }

   __global__ void generate_camera_rays_stratified_random_kernel(
         float3* const sample_origins,
         float3* const sample_directions,
         float3* const sample_directions_inverse,
         curandState_t* const rng_states,
         const unsigned int width,
         const float height,
         const float tan_fov_half,
         const int sample_num,
         const int total_num_samples) {
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;

      const float width_f = (float)width;
      const float pixel_x = (float) (pixel_index % width);
      const float pixel_y = (float) (pixel_index / width);

      const float num_samples_sqrt = sqrtf((float)total_num_samples);

      const float stratum_width_x = 1.f / (float)num_samples_sqrt;
      const float uniform_sample_x = curand_uniform(&rng_states[pixel_index]);
      const float scaled_uniform_sample_x = uniform_sample_x * stratum_width_x;
      const float stratum_offset_x = stratum_width_x * (float)(sample_num / num_samples_sqrt);
      const float stratified_sample_x = stratum_offset_x + scaled_uniform_sample_x;

      const float stratum_width_y = 1.f / (float)num_samples_sqrt;
      const float uniform_sample_y = curand_uniform(&rng_states[pixel_index]);
      const float scaled_uniform_sample_y = uniform_sample_y * stratum_width_y;
      const float stratum_offset_y = stratum_width_y * (float)(sample_num % (int)num_samples_sqrt);
      const float stratified_sample_y = stratum_offset_y + scaled_uniform_sample_y;
      
      const float x_offset = stratified_sample_x;
      const float y_offset = stratified_sample_y;

      const float aspect = width_f / height;

      // non-stratified
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
   
   __global__ void generate_camera_rays_random_kernel(float3 *const sample_origins, 
                                                      float3 *const sample_directions,
                                                      float3 *const sample_directions_inverse,
                                                      curandState_t *const rng_states,
                                                      const unsigned int width, 
                                                      const float height,
                                                      const float tan_fov_half,
                                                      const int sample_num, 
                                                      const int total_num_samples,
                                                      int device_index) {
      const unsigned int pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
      
      
      const float width_f = (float)width;
      const float pixel_x = (float) (pixel_index % width);
      const float pixel_y = (float) (pixel_index / width);

      const float num_samples_sqrt = sqrtf((float)total_num_samples);

      const float x_offset = curand_uniform(&rng_states[pixel_index]);
      const float y_offset = curand_uniform(&rng_states[pixel_index]);
      
      const float aspect = width_f / height;

      float offset_factor = 0;
      if (device_index == 1)
         offset_factor = 1;
      
      // non-stratified
      const float pixel_ndc_x = (pixel_x + x_offset) / (width_f) + (0.5f * offset_factor);
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
   
   void path_tracer::Trace(unsigned int num_samples) const {

      const float tan_fov_half = std::tan(device_context->camera_fov * poly::PIOver360);

      struct poly::device_pointers device_pointers {
            device_context->meshes,
            device_context->num_meshes,
            device_context->mesh_geometries,
            device_context->device_samples,
            device_context->device_bvh,
            device_context->num_bvh_nodes,
            device_context->index_pair,
            device_context->num_indices
      };

      cudaError_t error = cudaSuccess;
      
      cuda_check_error( cudaMemcpyToSymbol(camera_to_world_matrix, device_context->camera_to_world_matrix, sizeof(float) * 16));
      cuda_check_error( cudaMemcpyToSymbol(const_device_pointers, &device_pointers, sizeof(struct device_pointers)) );
      cuda_check_error( cudaMemcpyToSymbol(const_bvh_nodes, device_context->scene_field->bvh_root.compact_root->nodes, sizeof(poly::device_bvh_node) * num_constant_nodes));
      
      const unsigned int blocksPerGrid = (device_context->pixel_count + threads_per_block - 1) / threads_per_block;

      curandState_t* device_states;
      cuda_check_error( cudaMalloc((void**)&device_states, sizeof(curandState_t) * device_context->pixel_count));
      {
         const unsigned int curand_init_tpb = 256;
         const unsigned int curand_init_bgp = (device_context->pixel_count + curand_init_tpb - 1) / curand_init_tpb;

         //printf("launching init_curand_states_kernel<<<%i, %i>>>\n", curand_init_bgp, curand_init_tpb);
         init_curand_states_kernel<<<curand_init_bgp, curand_init_tpb>>>(device_states, device_context->pixel_count);
         cudaDeviceSynchronize();
         error = cudaGetLastError();
   
         if (error != cudaSuccess)
         {
            LOG_ERROR("Kernel init_curand_states_kernel failed with error: " << cudaGetErrorString(error));
            exit(EXIT_FAILURE);
         }
      }
      
      const unsigned int width = device_context->width;
      const unsigned int height = device_context->height;

      const size_t sample_bytes = sizeof(float3) * width * height;
      
      float3* sample_origins;
      float3* sample_directions;
      float3* sample_directions_inverse;
      
      cuda_check_error( cudaMalloc((void**)&sample_origins, sample_bytes) );
      cuda_check_error( cudaMalloc((void**)&sample_directions, sample_bytes) );
      cuda_check_error( cudaMalloc((void**)&sample_directions_inverse, sample_bytes) );

      for (int i = 0; i < num_samples; i++) {
         const unsigned int generate_rays_tpb = 64;
         const unsigned int generate_rays_bpg = (device_context->pixel_count + generate_rays_tpb - 1) / generate_rays_tpb;
         
         // generate rays
         generate_camera_rays_random_kernel<<<generate_rays_bpg, generate_rays_tpb>>>(
               sample_origins, 
               sample_directions, 
               sample_directions_inverse, 
               device_states,
               width, 
               (float) height, 
               tan_fov_half, 
               i,
               num_samples, 
               device_context->device_index);

//            generate_camera_rays_stratified_random_kernel<<<generate_rays_bpg, generate_rays_tpb>>>(
//                  sample_origins, sample_directions, sample_directions_inverse, device_states,
//                  width, (float) height, tan_fov_half, i, num_samples);

//            generate_camera_rays_centered_kernel<<<generate_rays_bpg, generate_rays_tpb>>>(
//                  sample_origins, sample_directions, sample_directions_inverse, 
//                  width, (float) height, tan_fov_half);
//            cudaDeviceSynchronize();
//            error = cudaGetLastError();
//
//            if (error != cudaSuccess) {
//               fprintf(stderr, "Failed to launch generate_camera_rays_centered_kernel (error code %s)!\n",
//                       cudaGetErrorString(error));
//               exit(EXIT_FAILURE);
//            }
         

         // run path tracer => sample_results

         //printf("launching path_trace_kernel<<<%i, %i>>>\n", blocksPerGrid, threads_per_block);
         path_trace_kernel<<<blocksPerGrid, threads_per_block>>>(
               sample_origins,
               sample_directions,
               sample_directions_inverse,
               device_states
         );
         

//         cudaDeviceSynchronize();
//         error = cudaGetLastError();
//
//         if (error != cudaSuccess) {
//            fprintf(stderr, "Failed to launch path_trace_kernel (error code %s)!\n", cudaGetErrorString(error));
//            exit(EXIT_FAILURE);
//         }
      }

      
      // consolidate samples

      const float num_samples_inv = 1.f / (float)num_samples;
      //printf("launching reduce_samples_kernel<<<%i, %i>>>\n", blocksPerGrid, threads_per_block);
      reduce_samples_kernel<<<blocksPerGrid, threads_per_block>>>(num_samples_inv, device_context->pixel_count);

      cudaDeviceSynchronize();
      error = cudaGetLastError();

      if (error != cudaSuccess) {
         LOG_ERROR("Kernel reduce_samples_kernel failed with error: " << cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      cuda_check_error( cudaFree(sample_origins) );
      cuda_check_error( cudaFree(sample_directions) );
      cuda_check_error( cudaFree(sample_directions_inverse) );
      cuda_check_error( cudaFree(device_states) );
   }
}
