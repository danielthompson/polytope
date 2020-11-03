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
      float bb[6];

      // interior - high child offset
      // leaf - face index offset
      // 4 bytes
      unsigned int offset;

      // 0 for interior, >0 for leaf
      // 2 bytes
      // TODO warn during BVH construction if the number of faces exceeds sizeof(unsigned short)
      unsigned short num_faces;
      
      // right now, just splitting axis
      // 2 bytes
      unsigned short flags;

      __host__ __device__ inline bool is_leaf() const {
         return num_faces > 0;
      }
   };
      
   struct device_mesh_geometry {
      // vertices
      float* x;
      float* y;
      float* z;

      // vertex normals
      float* nx;
      float* ny;
      float* nz;

      size_t num_vertices, num_faces, num_bytes;
      bool has_vertex_normals = false;

   };
   
   struct DeviceMesh {
      float obj_to_world[16];
      float world_to_object[16];
      float brdf_params[10];
      float world_bb[6];
      
      /**
       * Index into GPUMemoryManager::mesh_geometries for this mesh's geometry
       */
      size_t device_mesh_geometry_offset;
      
      poly::BRDF_TYPE brdf_type;
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
      struct device_mesh_geometry* mesh_geometries;
      
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
