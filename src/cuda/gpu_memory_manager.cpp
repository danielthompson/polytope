//
// Created by daniel on 5/15/20.
//

#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include "gpu_memory_manager.h"
#include "check_error.h"
#include "../cpu/shading/brdf/lambert_brdf.h"
#include "../cpu/shading/brdf/mirror_brdf.h"

namespace poly {

//   __constant__ float camera_to_world_matrix[16];
   
   size_t GPUMemoryManager::MallocScene(poly::Scene *scene) {

      this->scene_field = scene;
      
      size_t bytes_copied = 0;
      
      cuda_check_error( cudaSetDevice(1) );
      
      if (scene->Camera == nullptr) {
         Log.warning("Warning: scene has no camera, using identity matrix with FOV = 50.");
         poly::Transform identity;
         memcpy(camera_to_world_matrix, identity.Matrix.Matrix, 16 * sizeof(float));
         camera_fov = 50;
      }
      else {
         memcpy(camera_to_world_matrix, scene->Camera->CameraToWorld.Matrix.Matrix, 16 * sizeof(float));
         camera_fov = scene->Camera->Settings.FieldOfView;
      }
      
      // samples
      
      cuda_check_error( cudaMalloc((void **)&(host_samples.r), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_samples.r);
      cuda_check_error( cudaMalloc((void **)&(host_samples.g), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_samples.g);
      cuda_check_error( cudaMalloc((void **)&(host_samples.b), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_samples.b);
      
      device_samples = nullptr;
      cuda_check_error( cudaMalloc((void **)&(device_samples), sizeof(struct Samples)) );
      to_free_list.push_back(device_samples);
      cuda_check_error( cudaMemcpy(device_samples, &host_samples, sizeof(struct Samples), cudaMemcpyHostToDevice) );

      bytes_copied += sizeof(struct Samples);
      
      // meshes
      constexpr size_t device_mesh_size = sizeof(struct DeviceMesh);
      
      struct DeviceMesh* host_meshes_temp = nullptr;
      size_t device_meshes_size = device_mesh_size * scene->Shapes.size();
      cuda_check_error( cudaMalloc((void **)&(host_meshes_temp), device_meshes_size) );
      
      meshes = host_meshes_temp;
      num_meshes = scene->Shapes.size();
      size_t offset = 0;
      
      // TODO try creating one stream per mesh and copying them async
      
      for (auto original_mesh : scene->Shapes) {

         assert(original_mesh != nullptr);
         
         const poly::Mesh* host_mesh = reinterpret_cast<const poly::Mesh *>(original_mesh);
         const std::shared_ptr<poly::mesh_geometry> geometry = host_mesh->mesh_geometry;
         
         assert(host_mesh != nullptr);
         assert(geometry != nullptr);
         assert(host_mesh->mesh_geometry->x.size() == host_mesh->mesh_geometry->y.size());
         assert(host_mesh->mesh_geometry->y.size() == host_mesh->mesh_geometry->z.size());

         const size_t num_vertices = host_mesh->mesh_geometry->x.size();
         const size_t num_faces = num_vertices / 3;
         const size_t num_bytes = sizeof(float) * num_vertices;
         
         // geometry
         
         struct device_mesh_geometry device_mesh_geometry_temp { };
         device_mesh_geometry_temp.nx = nullptr;
         device_mesh_geometry_temp.ny = nullptr;
         device_mesh_geometry_temp.nz = nullptr;
         device_mesh_geometry_temp.num_vertices = num_vertices;
         device_mesh_geometry_temp.num_faces = num_faces;
         
         struct DeviceMesh host_mesh_temp { };
         host_mesh_temp.num_bytes = num_bytes;
         if (host_mesh->material) {
            host_mesh_temp.brdf_type = host_mesh->material->BRDF->brdf_type;
            switch (host_mesh_temp.brdf_type) {
               case (BRDF_TYPE::Lambert): {
                  poly::LambertBRDF* brdf = dynamic_cast<poly::LambertBRDF*>(&(*(host_mesh->material->BRDF)));
                  host_mesh_temp.brdf_params[0] = brdf->refl.r;
                  host_mesh_temp.brdf_params[1] = brdf->refl.g;
                  host_mesh_temp.brdf_params[2] = brdf->refl.b;
                  break;
               }
               case (BRDF_TYPE::Mirror): {
                  poly::MirrorBRDF* brdf = dynamic_cast<poly::MirrorBRDF*>(&(*(host_mesh->material->BRDF)));
                  host_mesh_temp.brdf_params[0] = brdf->refl.r;
                  host_mesh_temp.brdf_params[1] = brdf->refl.g;
                  host_mesh_temp.brdf_params[2] = brdf->refl.b;
                  break;
               }
            }
         }
         else {
            host_mesh_temp.brdf_type = BRDF_TYPE::None;
         }
         
         host_mesh_temp.aabb[0] = host_mesh->world_bb.p0.x;
         host_mesh_temp.aabb[1] = host_mesh->world_bb.p0.y;
         host_mesh_temp.aabb[2] = host_mesh->world_bb.p0.z;
         host_mesh_temp.aabb[3] = host_mesh->world_bb.p1.x;
         host_mesh_temp.aabb[4] = host_mesh->world_bb.p1.y;
         host_mesh_temp.aabb[5] = host_mesh->world_bb.p1.z;
         
         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.x), num_bytes) );
         to_free_list.push_back(host_mesh_temp.x);
         cuda_check_error( cudaMemcpy(host_mesh_temp.x, &(host_mesh->mesh_geometry->x[0]), num_bytes, cudaMemcpyHostToDevice) );

         bytes_copied += num_bytes;
         
         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.y), num_bytes) );
         to_free_list.push_back(host_mesh_temp.y);
         cuda_check_error( cudaMemcpy(host_mesh_temp.y, &(host_mesh->mesh_geometry->y[0]), num_bytes, cudaMemcpyHostToDevice) );

         bytes_copied += num_bytes;
         
         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.z), num_bytes) );
         to_free_list.push_back(host_mesh_temp.z);
         cuda_check_error( cudaMemcpy(host_mesh_temp.z, &(host_mesh->mesh_geometry->z[0]), num_bytes, cudaMemcpyHostToDevice) );

         bytes_copied += num_bytes;

         if (host_mesh->mesh_geometry->has_vertex_normals) {
            host_mesh_temp.has_vertex_normals = true;
            cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.nx), num_bytes) );
            to_free_list.push_back(host_mesh_temp.nx);
            cuda_check_error( cudaMemcpy(host_mesh_temp.nx, &(host_mesh->mesh_geometry->nx[0]), num_bytes, cudaMemcpyHostToDevice) );

            bytes_copied += num_bytes;

            cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.ny), num_bytes) );
            to_free_list.push_back(host_mesh_temp.ny);
            cuda_check_error( cudaMemcpy(host_mesh_temp.ny, &(host_mesh->mesh_geometry->ny[0]), num_bytes, cudaMemcpyHostToDevice) );

            bytes_copied += num_bytes;

            cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.nz), num_bytes) );
            to_free_list.push_back(host_mesh_temp.nz);
            cuda_check_error( cudaMemcpy(host_mesh_temp.nz, &(host_mesh->mesh_geometry->nz[0]), num_bytes, cudaMemcpyHostToDevice) );

            bytes_copied += num_bytes;
         }
         
         // TODO material / BRDF
//         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.src), sizeof(float) * 3) );
//         cuda_check_error( cudaMemcpy(host_mesh_temp.src, &(host_mesh->src[0]), num_bytes, cudaMemcpyHostToDevice) );
                                                                                                                                                                     
         cuda_check_error( cudaMemcpy(host_meshes_temp + offset, &host_mesh_temp, device_mesh_size, cudaMemcpyHostToDevice));

         bytes_copied += device_mesh_size;
         
         offset++;
      }
      to_free_list.push_back(host_meshes_temp);
      
      // bvh
      size_t num_bvh_bytes = scene->bvh_root.num_nodes * sizeof(device_bvh_node);
      cuda_check_error( cudaMalloc((void**)&device_bvh, num_bvh_bytes) );
      cuda_check_error( cudaMemcpy(device_bvh, scene->bvh_root.compact_root->nodes, num_bvh_bytes, cudaMemcpyHostToDevice) );
      
      
      device_bvh_node* gpu_check_nodes = static_cast<device_bvh_node *>(malloc(num_bvh_bytes));
      cuda_check_error( cudaMemcpy(gpu_check_nodes, device_bvh, num_bvh_bytes, cudaMemcpyDeviceToHost) );

      num_indices = scene->bvh_root.compact_root->leaf_ordered_indices.size();

      for (int i = 0; i < num_indices; i++) {
         device_bvh_node gpu_node = gpu_check_nodes[i];
         compact_bvh_node cpu_node = scene->bvh_root.compact_root->nodes[i];
         
         assert(gpu_node.is_leaf() == cpu_node.is_leaf());
         assert(gpu_node.num_faces == cpu_node.num_faces);
         if (cpu_node.is_leaf()) {
            assert(gpu_node.offset == cpu_node.face_index_offset);   
         }
         else {
            assert(gpu_node.offset == cpu_node.high_child_offset);
         }
         assert(gpu_node.flags == cpu_node.flags);
         assert(gpu_node.aabb[0] == cpu_node.bb.p0.x);
         assert(gpu_node.aabb[1] == cpu_node.bb.p0.y);
         assert(gpu_node.aabb[2] == cpu_node.bb.p0.z);
         assert(gpu_node.aabb[3] == cpu_node.bb.p1.x);
         assert(gpu_node.aabb[4] == cpu_node.bb.p1.y);
         assert(gpu_node.aabb[5] == cpu_node.bb.p1.z);
      }
      
      bytes_copied += num_bvh_bytes;
      to_free_list.push_back(device_bvh);
      
      const size_t num_index_bytes = scene->bvh_root.compact_root->leaf_ordered_indices.size() * sizeof(scene->bvh_root.compact_root->leaf_ordered_indices);
      cuda_check_error( cudaMalloc((void**)&index_pair, num_index_bytes) );
      cuda_check_error( cudaMemcpy(index_pair, &scene->bvh_root.compact_root->leaf_ordered_indices[0], num_index_bytes, cudaMemcpyHostToDevice) );

      bytes_copied += num_index_bytes;
      
      free(gpu_check_nodes);
      return bytes_copied;
   }
   
   GPUMemoryManager::~GPUMemoryManager() {
      for (void* ptr : to_free_list) {
         cuda_check_error( cudaFree(ptr) );
      }
   }
}
