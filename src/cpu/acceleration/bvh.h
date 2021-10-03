
//
// Created by daniel on 6/9/20.
//

#ifndef POLY_BVH_H
#define POLY_BVH_H

#include "../structures/bounding_box.h"
#include "../shapes/mesh.h"
#include "../../common/utilities/Common.h"

namespace poly {
   class bvh_node {
   public:
      // TODO tighten this up
      bvh_node* high; // interior
      bvh_node* low; // interior
      poly::axis axis; // interior
      poly::bounding_box bb; // both
      std::vector<std::pair<unsigned int, unsigned int>> indices; // leaf
   };
   
   class compact_bvh_node {
   public:
      // if we can get this down to 32 bytes, and cache line size is 64 bytes, and we alloc the containing array on a 64 byte
      // boundary, then we can get exactly 2 nodes per cache line, which will result in a cache hit for the case in which 
      // a parent is at the first 32 bytes and its first child is at the 2nd 32 bytes
      
      poly::bounding_box bb; // 24 bytes (both)
      // TODO - at build time, sort face indices by the order in which they are put into the tree
      // then, this can just be a 4-byte offset, instead of an 8-byte pointer
      union {
         unsigned int high_child_offset; // only for interior 
         unsigned int face_index_offset; // only for leaf
      }; // 4 bytes
      
      unsigned short num_faces; // 0 for interior, >0 for leaf (2 bytes) 
      unsigned short flags; // both (2 bytes)
      
      inline bool is_leaf() const {
         return num_faces > 0;
      }
      
      inline poly::axis get_axis() const {
         return (poly::axis)flags;
      }
      
      inline void set_axis(poly::axis axis) {
         flags = (short)axis;
      }

   };
   
   class compact_bvh {
   public:
      compact_bvh(unsigned int num_nodes) : num_nodes(num_nodes){
         LOG_DEBUG("sizeof compact_bvh_node: " << sizeof(compact_bvh_node));
//         nodes = static_cast<compact_bvh_node *>(malloc(sizeof(compact_bvh_node) * num_nodes));
         
         size_t size = sizeof(compact_bvh_node) * num_nodes;

         const int remainder = size % 64;
         if (remainder > 0) 
         {
            size += (64 - remainder);
         }
         
         nodes = static_cast<compact_bvh_node *>(aligned_alloc(64, size));
         if (nodes == nullptr) {
            ERROR( "compact_bvh: couldn't get aligned address :/");
         }
         
         leaf_ordered_indices.reserve(num_nodes);
      }
      
      ~compact_bvh() {
         if (nodes)
            free(nodes);
      }
      
      std::vector<std::pair<unsigned int, unsigned int>> leaf_ordered_indices;
      
      compact_bvh_node* nodes;
      unsigned int num_nodes;
   };
   
   class bvh {
   public:
      bvh() : compact_root(nullptr), root(nullptr), num_nodes(0) { }
      ~bvh();
      unsigned int bound(const std::vector<poly::Mesh*>& meshes);
      void compact();
      void metrics() const;
      bool hits(const poly::ray &ray) const;
      bool hits_compact(const poly::ray &ray) const;
      void intersect(poly::ray &ray, poly::intersection& intersection) const;
      poly::intersection intersect_compact(poly::ray &ray, int x, int y) const;
      bvh_node* root;
      compact_bvh* compact_root;
      std::vector<poly::Mesh*> meshes;
      unsigned int num_nodes;
      std::vector<std::pair<unsigned int, unsigned int>> master_indices;
   private:
      void compact_helper_iterative(bvh_node* root);
      unsigned int compact_helper(bvh_node* node, unsigned int index);
   };
   
}

#endif //POLY_BVH_H
