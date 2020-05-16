//
// Created by daniel on 5/15/20.
//

#include <cuda_runtime.h>
#include <cassert>
#include "gpu_memory_manager.h"

namespace Polytope {
   void GPUMemoryManager::AddMesh(Polytope::MeshLinearSOA *mesh) {

      assert(mesh != nullptr);
      assert(mesh->x.size() == mesh->y.size());
      assert(mesh->y.size() == mesh->z.size());
      
      size_t num_bytes = sizeof(float) * mesh->x.size();
      
      float* d_x = nullptr;
      cudaError_t error = cudaMalloc((void **)&d_x, num_bytes);
      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to malloc device memory for mesh x coords (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }

      float* d_y = nullptr;
      error = cudaMalloc((void **)&d_y, num_bytes);
      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to malloc device memory for mesh y coords (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }

      float* d_z = nullptr;
      error = cudaMalloc((void **)&d_z, num_bytes);
      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to malloc device memory for mesh z coords (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      meshes_on_device.push_back(std::make_shared<DeviceMesh>(d_x, d_y, d_z, num_bytes, mesh));

      size_t offset = 0;
      
      error = cudaMemcpy(d_x, &(mesh->x[0]), num_bytes, cudaMemcpyHostToDevice);
      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to copy mesh x coords to device (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }

      error = cudaMemcpy(d_y, &(mesh->y[0]), num_bytes, cudaMemcpyHostToDevice);
      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to copy mesh y coords to device (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }

      error = cudaMemcpy(d_z, &(mesh->z[0]), num_bytes, cudaMemcpyHostToDevice);
      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to copy mesh z coords to device (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
   }


   std::shared_ptr<CameraRays> GPUMemoryManager::MallocCameraRays() {
      constexpr unsigned int width = 640;
      constexpr unsigned int height = 480;
      constexpr unsigned int num_pixels = width * height;

      cudaError_t error = cudaSuccess;

      camera_rays = std::make_shared<CameraRays>();

      for (int i = 0; i < 3; i++) {
         error = cudaMalloc((void **)&(camera_rays->d_o[i]), sizeof(float) * num_pixels);
         if (error != cudaSuccess) {
            fprintf(stderr, "Failed to malloc ray origin %i", i);
            exit(EXIT_FAILURE);
         }
      };

      for (int i = 0; i < 3; i++) {
         error = cudaMalloc((void **)&(camera_rays->d_d[i]), sizeof(float) * num_pixels);
         if (error != cudaSuccess) {
            fprintf(stderr, "Failed to malloc ray direction (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
         }
      };

      return camera_rays;
   }

   void GPUMemoryManager::MallocSamples() {
      constexpr unsigned int width = 640;
      constexpr unsigned int height = 480;
      constexpr unsigned int num_pixels = width * height;

      cudaError_t error = cudaSuccess;
      error = cudaMalloc((void **)&d_samples, sizeof(float) * num_pixels);
      if (error != cudaSuccess) {
         fprintf(stderr, "Failed to malloc for samples on device (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
   }
   
   GPUMemoryManager::~GPUMemoryManager() {
      for (const auto &device_mesh : meshes_on_device) {
         if (device_mesh != nullptr) {
            cudaFree(device_mesh->d_x);
            cudaFree(device_mesh->d_y);
            cudaFree(device_mesh->d_z);   
         }
      }
      
      if (camera_rays != nullptr) {
         cudaFree(camera_rays->d_o[0]);
         cudaFree(camera_rays->d_o[1]);
         cudaFree(camera_rays->d_o[2]);
         cudaFree(camera_rays->d_d[0]);
         cudaFree(camera_rays->d_d[1]);
         cudaFree(camera_rays->d_d[2]);
      }
   }
}
