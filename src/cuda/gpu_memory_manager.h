//
// Created by daniel on 5/15/20.
//

#ifndef POLY_GPU_MEMORY_MANAGER_H
#define POLY_GPU_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include "../cpu/shapes/mesh.h"
#include "../cpu/scenes/Scene.h"

namespace poly {
   
   struct DeviceMesh {
      float* x;
      float* y;
      float* z;
      
      // color.. need to fix this to properly use brdf
      float* src;
      size_t num_bytes, num_vertices, num_faces;
      float aabb[6];
   };

   struct Samples {
      float* r;
      float* g;
      float* b;
   };
   
   class GPUMemoryManager {
   public:
      GPUMemoryManager(const unsigned int width, const unsigned int height) 
      : width(width), height(height), device_camera(nullptr), meshes(nullptr) { 
         num_pixels = width * height;
      }
      ~GPUMemoryManager();
      
      size_t MallocScene(poly::Scene* scene);
      struct DeviceCamera* device_camera;
      struct DeviceMesh* meshes;
      unsigned int num_meshes;
      
      float camera_to_world_matrix[16];
      float camera_fov;
      
      struct Samples* device_samples;
      struct Samples host_samples;
      
      unsigned int width, height, num_pixels;
      
      std::vector<void *> to_free_list;
   };

}

#endif //POLY_GPU_MEMORY_MANAGER_H
