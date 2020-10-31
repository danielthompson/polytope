//
// Created by daniel on 5/15/20.
//

#ifndef POLY_GPU_MEMORY_MANAGER_H
#define POLY_GPU_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include "../cpu/shapes/mesh.h"
#include "../cpu/scenes/Scene.h"

namespace poly {
   
   struct device_index_pair {
      unsigned int mesh_index;
      unsigned int face_index;
   };
   
   struct device_bvh_node {
      // 24 bytes
      float aabb[6];

      // interior - high child offset
      // leaf - face index offset
      // 4 bytes
      unsigned int offset;

      // 0 for interior, >0 for leaf
      // 2 bytes
      unsigned short num_faces;
      
      // right now, just splitting axis
      // 2 bytes
      unsigned short flags;

      __host__ __device__ inline bool is_leaf() const {
         return num_faces > 0;
      }
   };
      
   struct device_mesh_geometry {
      float* x;
      float* y;
      float* z;

      float* nx;
      float* ny;
      float* nz;

      bool has_vertex_normals = false;
      size_t num_vertices, num_faces;
   };
   
   struct DeviceMesh {
      
      poly::BRDF_TYPE brdf_type;
      float brdf_params[10];
      
      size_t num_bytes;
      float aabb[6];
      
      float obj_to_world[16];

      // TODO remove this stuff:
      float* x;
      float* y;
      float* z;

      float* nx;
      float* ny;
      float* nz;

      bool has_vertex_normals = false;
      size_t num_vertices, num_faces;
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
      
      poly::Scene* scene_field;
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
      
      device_bvh_node* device_bvh;
      unsigned int num_bvh_nodes;
      
      device_index_pair* index_pair;
      unsigned int num_indices;
   };

}

#endif //POLY_GPU_MEMORY_MANAGER_H
