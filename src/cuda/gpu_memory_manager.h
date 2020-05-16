//
// Created by daniel on 5/15/20.
//

#ifndef POLYTOPE_GPU_MEMORY_MANAGER_H
#define POLYTOPE_GPU_MEMORY_MANAGER_H

#include "../cpu/shapes/linear_soa/mesh_linear_soa.h"

namespace Polytope {

   struct DeviceMesh {
      DeviceMesh(float *d_x, float* d_y, float* d_z, size_t bytes, Polytope::MeshLinearSOA* host_mesh) 
        : d_x(d_x), d_y(d_y), d_z(d_z), bytes(bytes), host_mesh(host_mesh) { }

      float* d_x;
      float* d_y;
      float* d_z;
      size_t bytes;
      Polytope::MeshLinearSOA* host_mesh;
   };
   
   struct CameraRays {
      float* d_o[3];
      float* d_d[3];
      size_t num_pixels;
   };
   
   class GPUMemoryManager {
   public:
      ~GPUMemoryManager();
      std::shared_ptr<CameraRays> MallocCameraRays();
      
      // TODO
      void MallocSamples();
      
      void AddMesh(Polytope::MeshLinearSOA* mesh);
      std::vector<std::shared_ptr<DeviceMesh>> meshes_on_device;
      std::shared_ptr<CameraRays> camera_rays;
      
      // device sample array
      float* d_samples;
      
      // device camera matrix
      float* d_cm;
   };

}


#endif //POLYTOPE_GPU_MEMORY_MANAGER_H
