//
// Created by daniel on 5/15/20.
//

#ifndef POLYTOPE_GPU_MEMORY_MANAGER_H
#define POLYTOPE_GPU_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include "../cpu/shapes/linear_soa/mesh_linear_soa.h"
#include "../cpu/scenes/AbstractScene.h"

namespace Polytope {
   
   struct DeviceCamera {
      // origin
      float* ox;
      float* oy;
      float* oz;
      
      // direction
      float* dx;
      float* dy;
      float* dz;
      
      // camera matrix
      float* cm;
      
      float fov;
      size_t num_pixels;
   };
   
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
      
      void MallocScene(Polytope::AbstractScene* scene);
      struct DeviceCamera* device_camera;
      struct DeviceMesh* meshes;
      unsigned int num_meshes;
      
      float camera_to_world_matrix[16];
      
      struct Samples* device_samples;
      struct Samples host_samples;
      
      unsigned int width, height, num_pixels;
      
      std::vector<void *> to_free_list;
   };

}


#endif //POLYTOPE_GPU_MEMORY_MANAGER_H
