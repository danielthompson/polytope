
//
// Created by daniel on 6/9/20.
//

#ifndef POLY_BVH_H
#define POLY_BVH_H

#include "../structures/BoundingBox.h"
#include "../shapes/mesh.h"
#include "../../common/utilities/Common.h"

namespace poly {
   class bvh_node {
   public:
      // TODO tighten this up
      bvh_node* high; // interior
      bvh_node* low; // interior
      //poly::Axis axis; // interior
      poly::BoundingBox bb; // both
      std::vector<unsigned int> face_indices; // leaf
   };
   
   class compact_bvh_node {
   public:
      poly::BoundingBox bb; // 24 bytes (both)
//      std::vector<unsigned int> face_indices; // 24 bytes
      // TODO - at build time, sort face indices by the order in which they are put into the tree
      // then, this can just be a 4-byte offset, instead of an 8-byte pointer
      unsigned int* face_indices; // 8 bytes (interior if nullptr)
      union {
         int high_offset; // only for interior 
         unsigned int num_face_indices; // only for leaf
      }; // 4 bytes
   };
   
   class compact_bvh {
   public:
      compact_bvh(unsigned int num_nodes) : num_nodes(num_nodes){
         nodes = new compact_bvh_node [num_nodes];
      }
      
      ~compact_bvh() {
         if (nodes)
            delete[] nodes;
      }
      
      compact_bvh_node* nodes;
      unsigned int num_nodes;
   };
   
   class bvh {
   public:
      bvh() = default;
      ~bvh();
      unsigned int bound(poly::Mesh* mesh);
      void compact();
      void metrics() const;
      bool hits(const poly::Ray &ray) const;
      bool hits_compact(const poly::Ray &ray) const;
      void intersect(poly::Ray &ray, poly::Intersection& intersection) const;
      void intersect_compact(poly::Ray &ray, poly::Intersection& intersection) const;
      bvh_node* root;
      compact_bvh* compact_root;
      poly::Mesh* single_mesh;
      unsigned int num_nodes;
   private:
      unsigned int compact_helper(bvh_node* node, unsigned int index);
   };
   
}

#endif //POLY_BVH_H
