//
// Created by daniel on 5/15/20.
//

#include <cuda_runtime.h>
#include <cassert>
#include "gpu_memory_manager.h"
#include "kernels/common_device_functions.cuh"
#include "check_error.h"
namespace Polytope {

   void GPUMemoryManager::MallocScene(Polytope::AbstractScene *scene) {

      // camera
      struct DeviceCamera host_camera_temp { };
      cuda_check_error( cudaMalloc((void **)&(host_camera_temp.ox), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_camera_temp.ox);
      cuda_check_error( cudaMalloc((void **)&(host_camera_temp.oy), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_camera_temp.oy);
      cuda_check_error( cudaMalloc((void **)&(host_camera_temp.oz), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_camera_temp.oz);
      cuda_check_error( cudaMalloc((void **)&(host_camera_temp.dx), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_camera_temp.dx);
      cuda_check_error( cudaMalloc((void **)&(host_camera_temp.dy), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_camera_temp.dy);
      cuda_check_error( cudaMalloc((void **)&(host_camera_temp.dz), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_camera_temp.dz);
      cuda_check_error( cudaMalloc((void **)&(host_camera_temp.cm), sizeof(float) * 16) );
      to_free_list.push_back(host_camera_temp.cm);
      host_camera_temp.num_pixels = num_pixels;
      // data will be provided by camera method

      device_camera = nullptr;
      cuda_check_error( cudaMalloc((void **)&(device_camera), sizeof(struct DeviceCamera)) );
      to_free_list.push_back(device_camera);
      cuda_check_error( cudaMemcpy(device_camera, &host_camera_temp, sizeof(struct DeviceCamera), cudaMemcpyHostToDevice) );
      
      // samples
      struct DeviceSamples host_samples_temp { };
      cuda_check_error( cudaMalloc((void **)&(host_samples_temp.r), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_samples_temp.r);
      cuda_check_error( cudaMalloc((void **)&(host_samples_temp.g), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_samples_temp.g);
      cuda_check_error( cudaMalloc((void **)&(host_samples_temp.b), sizeof(float) * num_pixels) );
      to_free_list.push_back(host_samples_temp.b);
      
      device_samples = nullptr;
      cuda_check_error( cudaMalloc((void **)&(device_samples), sizeof(struct DeviceSamples)) );
      to_free_list.push_back(device_samples);
      cuda_check_error( cudaMemcpy(device_samples, &host_samples_temp, sizeof(struct DeviceSamples), cudaMemcpyHostToDevice) );
      
      // meshes
      constexpr size_t device_mesh_size = sizeof(struct DeviceMesh);
      
      struct DeviceMesh* host_meshes_temp = nullptr;
      cuda_check_error( cudaMalloc((void **)&(host_meshes_temp), device_mesh_size * scene->Shapes.size()) );
      
      size_t offset = 0;
      
      for (unsigned int i = 0; i < scene->Shapes.size(); i++) {

         const auto original_mesh = scene->Shapes.at(i);
         assert(original_mesh != nullptr);
         
         const Polytope::MeshLinearSOA* host_mesh = reinterpret_cast<const Polytope::MeshLinearSOA *>(original_mesh);
         assert(host_mesh != nullptr);
         assert(host_mesh->x.size() == host_mesh->y.size());
         assert(host_mesh->y.size() == host_mesh->z.size());

         const size_t num_vertices = host_mesh->x.size();
         const size_t num_faces = num_vertices / 3;
         const size_t num_bytes = sizeof(float) * num_vertices;
         
         struct DeviceMesh host_mesh_temp { };
         host_mesh_temp.num_bytes = num_bytes;
         host_mesh_temp.num_vertices = num_vertices;
         host_mesh_temp.num_faces = num_faces;
         
         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.x), num_bytes) );
         to_free_list.push_back(host_mesh_temp.x);
         cuda_check_error( cudaMemcpy(host_mesh_temp.x, &(host_mesh->x[0]), num_bytes, cudaMemcpyHostToDevice) );
         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.y), num_bytes) );
         to_free_list.push_back(host_mesh_temp.y);
         cuda_check_error( cudaMemcpy(host_mesh_temp.y, &(host_mesh->y[0]), num_bytes, cudaMemcpyHostToDevice) );
         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.z), num_bytes) );
         to_free_list.push_back(host_mesh_temp.z);
         cuda_check_error( cudaMemcpy(host_mesh_temp.z, &(host_mesh->z[0]), num_bytes, cudaMemcpyHostToDevice) );

         // TODO material / BRDF
//         cuda_check_error( cudaMalloc((void **)&(host_mesh_temp.src), sizeof(float) * 3) );
//         cuda_check_error( cudaMemcpy(host_mesh_temp.src, &(host_mesh->src[0]), num_bytes, cudaMemcpyHostToDevice) );
         
         cuda_check_error( cudaMemcpy(host_meshes_temp + offset, &host_mesh_temp, device_mesh_size, cudaMemcpyHostToDevice));
         offset += device_mesh_size;
      }

      to_free_list.push_back(host_meshes_temp);
      
   }
   
   
   GPUMemoryManager::~GPUMemoryManager() {
      for (void* ptr : to_free_list) {
         cuda_check_error( cudaFree(ptr) );
      }
   }
}
