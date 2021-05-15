//
// Created by daniel on 5/15/20.
//

#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <unordered_map>

#include "gpu_memory_manager.h"
#include "check_error.h"
#include "../cpu/shading/brdf/lambert_brdf.h"
#include "../cpu/shading/brdf/mirror_brdf.h"

namespace poly {

//   __constant__ float camera_to_world_matrix[16];
   
   size_t GPUMemoryManager::MallocScene(poly::Scene *scene) {

      this->scene_field = scene;
      
      size_t total_bytes_copied = 0;
      
      for (int device_index = 0; device_index < num_devices; device_index++) {
         device_free_lists.emplace_back();
         std::vector<void *>* device_free_list = &device_free_lists[device_index];
         
         size_t device_bytes_copied = 0;

         cuda_check_error(cudaSetDevice(device_index));

         if (scene->Camera == nullptr) {
            Log.warning("Warning: scene has no camera, using identity matrix with FOV = 50.");
            poly::Transform identity;
            memcpy(camera_to_world_matrix, identity.Matrix.Matrix, 16 * sizeof(float));
            camera_fov = 50;
         } else {
            memcpy(camera_to_world_matrix, scene->Camera->CameraToWorld.Matrix.Matrix, 16 * sizeof(float));
            camera_fov = scene->Camera->Settings.FieldOfView;
         }

         // samples

         cuda_check_error(cudaMalloc((void **) &(host_samples.r), sizeof(float) * num_pixels));
         device_free_list->push_back(host_samples.r);
         cuda_check_error(cudaMalloc((void **) &(host_samples.g), sizeof(float) * num_pixels));
         device_free_list->push_back(host_samples.g);
         cuda_check_error(cudaMalloc((void **) &(host_samples.b), sizeof(float) * num_pixels));
         device_free_list->push_back(host_samples.b);

         device_samples = nullptr;
         cuda_check_error(cudaMalloc((void **) &(device_samples), sizeof(struct Samples)));
         device_free_list->push_back(device_samples);
         cuda_check_error(cudaMemcpy(device_samples, &host_samples, sizeof(struct Samples), cudaMemcpyHostToDevice));

         device_bytes_copied += sizeof(struct Samples);


         // meshes
         constexpr size_t device_mesh_size = sizeof(struct DeviceMesh);

         struct DeviceMesh *host_meshes_temp = nullptr;
         size_t device_meshes_size = device_mesh_size * scene->Shapes.size();
         cuda_check_error(cudaMalloc((void **) &(host_meshes_temp), device_meshes_size));

         // TODO unit test to ensure scene->num_mesh_geometries is the right number
         struct device_mesh_geometry *device_mesh_geometries_temp = nullptr;
         size_t device_mesh_geometries_size = sizeof(struct device_mesh_geometry) * scene->num_mesh_geometries;
         cuda_check_error(cudaMalloc((void **) &(device_mesh_geometries_temp), device_mesh_geometries_size));
         this->mesh_geometries = device_mesh_geometries_temp;

         meshes = host_meshes_temp;
         num_meshes = scene->Shapes.size();
         size_t device_mesh_offset = 0;
         size_t next_device_mesh_geometry_index = 0;

         std::unordered_map<std::shared_ptr<poly::mesh_geometry>, size_t> already_copied;

         // TODO try creating one stream per mesh and copying them async
         for (auto original_mesh : scene->Shapes) {

            assert(original_mesh != nullptr);

            // TODO nuke this cast
            const poly::Mesh *host_mesh = reinterpret_cast<const poly::Mesh *>(original_mesh);
            const std::shared_ptr<poly::mesh_geometry> geometry = host_mesh->mesh_geometry;
            assert(host_mesh != nullptr);

            // if the geometry hasn't already been copied, copy it
            // TODO we need to put the offset in the hash table too!
            size_t device_mesh_geometry_index = next_device_mesh_geometry_index;
            try {
               device_mesh_geometry_index = already_copied.at(host_mesh->mesh_geometry);
            }
            catch (...) {
               // it hasn't been copied yet

               // 1. populate it
               assert(geometry != nullptr);
               assert(host_mesh->mesh_geometry->x.size() == host_mesh->mesh_geometry->y.size());
               assert(host_mesh->mesh_geometry->y.size() == host_mesh->mesh_geometry->z.size());

               const size_t num_vertices = host_mesh->mesh_geometry->x.size();
               const size_t num_faces = num_vertices / 3;
               const size_t num_bytes = sizeof(float) * num_vertices;

               struct device_mesh_geometry device_mesh_geometry_temp{};
               device_mesh_geometry_temp.nx = nullptr;
               device_mesh_geometry_temp.ny = nullptr;
               device_mesh_geometry_temp.nz = nullptr;
               device_mesh_geometry_temp.num_vertices = num_vertices;
               device_mesh_geometry_temp.num_faces = num_faces;
               device_mesh_geometry_temp.num_bytes = num_bytes;

               cuda_check_error(cudaMalloc((void **) &(device_mesh_geometry_temp.x), num_bytes));
               device_free_list->push_back(device_mesh_geometry_temp.x);
               cuda_check_error(cudaMemcpy(device_mesh_geometry_temp.x, &(host_mesh->mesh_geometry->x[0]), num_bytes,
                                           cudaMemcpyHostToDevice));

               device_bytes_copied += num_bytes;

               cuda_check_error(cudaMalloc((void **) &(device_mesh_geometry_temp.y), num_bytes));
               device_free_list->push_back(device_mesh_geometry_temp.y);
               cuda_check_error(cudaMemcpy(device_mesh_geometry_temp.y, &(host_mesh->mesh_geometry->y[0]), num_bytes,
                                           cudaMemcpyHostToDevice));

               device_bytes_copied += num_bytes;

               cuda_check_error(cudaMalloc((void **) &(device_mesh_geometry_temp.z), num_bytes));
               device_free_list->push_back(device_mesh_geometry_temp.z);
               cuda_check_error(cudaMemcpy(device_mesh_geometry_temp.z, &(host_mesh->mesh_geometry->z[0]), num_bytes,
                                           cudaMemcpyHostToDevice));

               device_bytes_copied += num_bytes;

               if (host_mesh->mesh_geometry->has_vertex_normals) {
                  device_mesh_geometry_temp.has_vertex_normals = true;
                  cuda_check_error(cudaMalloc((void **) &(device_mesh_geometry_temp.nx), num_bytes));
                  device_free_list->push_back(device_mesh_geometry_temp.nx);
                  cuda_check_error(
                        cudaMemcpy(device_mesh_geometry_temp.nx, &(host_mesh->mesh_geometry->nx[0]), num_bytes,
                                   cudaMemcpyHostToDevice));

                  device_bytes_copied += num_bytes;

                  cuda_check_error(cudaMalloc((void **) &(device_mesh_geometry_temp.ny), num_bytes));
                  device_free_list->push_back(device_mesh_geometry_temp.ny);
                  cuda_check_error(
                        cudaMemcpy(device_mesh_geometry_temp.ny, &(host_mesh->mesh_geometry->ny[0]), num_bytes,
                                   cudaMemcpyHostToDevice));

                  device_bytes_copied += num_bytes;

                  cuda_check_error(cudaMalloc((void **) &(device_mesh_geometry_temp.nz), num_bytes));
                  device_free_list->push_back(device_mesh_geometry_temp.nz);
                  cuda_check_error(
                        cudaMemcpy(device_mesh_geometry_temp.nz, &(host_mesh->mesh_geometry->nz[0]), num_bytes,
                                   cudaMemcpyHostToDevice));

                  device_bytes_copied += num_bytes;
               }

               // 2. copy it to the device
               cuda_check_error(cudaMemcpy(device_mesh_geometries_temp + next_device_mesh_geometry_index,
                                           &device_mesh_geometry_temp, sizeof(struct device_mesh_geometry),
                                           cudaMemcpyHostToDevice));
               device_mesh_geometry_index = next_device_mesh_geometry_index;

               // 3. add it to the hash table
               already_copied[host_mesh->mesh_geometry] = device_mesh_geometry_index;

               // 4. increment it for next time
               next_device_mesh_geometry_index++;
            }

            // if it has been copied, just use the existing device geometry offset
            struct DeviceMesh host_mesh_temp{};
            if (host_mesh->material) {
               host_mesh_temp.brdf_type = host_mesh->material->BRDF->brdf_type;
               switch (host_mesh_temp.brdf_type) {
                  case (BRDF_TYPE::Lambert): {
                     poly::LambertBRDF *brdf = dynamic_cast<poly::LambertBRDF *>(&(*(host_mesh->material->BRDF)));
                     host_mesh_temp.brdf_params[0] = brdf->refl.r;
                     host_mesh_temp.brdf_params[1] = brdf->refl.g;
                     host_mesh_temp.brdf_params[2] = brdf->refl.b;
                     break;
                  }
                  case (BRDF_TYPE::Mirror): {
                     poly::MirrorBRDF *brdf = dynamic_cast<poly::MirrorBRDF *>(&(*(host_mesh->material->BRDF)));
                     host_mesh_temp.brdf_params[0] = brdf->refl.r;
                     host_mesh_temp.brdf_params[1] = brdf->refl.g;
                     host_mesh_temp.brdf_params[2] = brdf->refl.b;
                     break;
                  }
               }
            } else {
               host_mesh_temp.brdf_type = BRDF_TYPE::None;
            }

            host_mesh_temp.world_bb[0] = host_mesh->world_bb.p0.x;
            host_mesh_temp.world_bb[1] = host_mesh->world_bb.p0.y;
            host_mesh_temp.world_bb[2] = host_mesh->world_bb.p0.z;
            host_mesh_temp.world_bb[3] = host_mesh->world_bb.p1.x;
            host_mesh_temp.world_bb[4] = host_mesh->world_bb.p1.y;
            host_mesh_temp.world_bb[5] = host_mesh->world_bb.p1.z;
            host_mesh_temp.device_mesh_geometry_offset = device_mesh_geometry_index;

            memcpy(host_mesh_temp.obj_to_world, host_mesh->object_to_world->Matrix.Matrix, 16 * sizeof(float));
            memcpy(host_mesh_temp.world_to_object, host_mesh->world_to_object->Matrix.Matrix, 16 * sizeof(float));

            // TODO material / BRDF
//         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.src), sizeof(float) * 3) );
//         cuda_check_error( cudaMemcpy(host_mesh_temp.src, &(host_mesh->src[0]), num_bytes, cudaMemcpyHostToDevice) );

            cuda_check_error(cudaMemcpy(host_meshes_temp + device_mesh_offset, &host_mesh_temp, device_mesh_size,
                                        cudaMemcpyHostToDevice));

            device_bytes_copied += device_mesh_size;

            device_mesh_offset++;
         }
         device_free_list->push_back(host_meshes_temp);

         // bvh
         size_t num_bvh_bytes = scene->bvh_root.num_nodes * sizeof(device_bvh_node);
         cuda_check_error(cudaMalloc((void **) &device_bvh, num_bvh_bytes));
         cuda_check_error(
               cudaMemcpy(device_bvh, scene->bvh_root.compact_root->nodes, num_bvh_bytes, cudaMemcpyHostToDevice));


         device_bvh_node *gpu_check_nodes = static_cast<device_bvh_node *>(malloc(num_bvh_bytes));
         cuda_check_error(cudaMemcpy(gpu_check_nodes, device_bvh, num_bvh_bytes, cudaMemcpyDeviceToHost));

         num_indices = scene->bvh_root.compact_root->leaf_ordered_indices.size();

//      for (int i = 0; i < num_indices; i++) {
//         device_bvh_node gpu_node = gpu_check_nodes[i];
//         compact_bvh_node cpu_node = scene->bvh_root.compact_root->nodes[i];
//         
//         assert(gpu_node.is_leaf() == cpu_node.is_leaf());
//         if (gpu_node.num_faces != cpu_node.num_faces) {
//            Log.error("BVH node %i: gpu has %i faces, cpu has %i", i, gpu_node.num_faces, cpu_node.num_faces);
//         }
//         assert(gpu_node.num_faces == cpu_node.num_faces);
//         if (cpu_node.is_leaf()) {
//            assert(gpu_node.offset == cpu_node.face_index_offset);   
//         }
//         else {
//            assert(gpu_node.offset == cpu_node.high_child_offset);
//         }
//         assert(gpu_node.flags == cpu_node.flags);
//         assert(gpu_node.bb[0] == cpu_node.bb.p0.x);
//         assert(gpu_node.bb[1] == cpu_node.bb.p0.y);
//         assert(gpu_node.bb[2] == cpu_node.bb.p0.z);
//         assert(gpu_node.bb[3] == cpu_node.bb.p1.x);
//         assert(gpu_node.bb[4] == cpu_node.bb.p1.y);
//         assert(gpu_node.bb[5] == cpu_node.bb.p1.z);
//      }

         device_bytes_copied += num_bvh_bytes;
         device_free_list->push_back(device_bvh);

         const size_t num_index_bytes = scene->bvh_root.compact_root->leaf_ordered_indices.size() *
                                        sizeof(scene->bvh_root.compact_root->leaf_ordered_indices);
         cuda_check_error(cudaMalloc((void **) &index_pair, num_index_bytes));
         cuda_check_error(
               cudaMemcpy(index_pair, &scene->bvh_root.compact_root->leaf_ordered_indices[0], num_index_bytes,
                          cudaMemcpyHostToDevice));

         device_bytes_copied += num_index_bytes;

         free(gpu_check_nodes);
         total_bytes_copied += device_bytes_copied;
      }
      return total_bytes_copied;
   }
   
   GPUMemoryManager::~GPUMemoryManager() {
      for (int device_index = 0; device_index < num_devices; device_index++) {
         cuda_check_error(cudaSetDevice(device_index));
         for (void *ptr : device_free_lists[device_index]) {
            cuda_check_error(cudaFree(ptr));
         }
      }
   }
}
