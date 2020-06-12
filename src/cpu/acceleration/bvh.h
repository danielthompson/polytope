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
      poly::Axis axis; // interior
      poly::BoundingBox bb; // both
      std::vector<unsigned int> face_indices; // leaf
   };
   
   class bvh {
   public:
      
      ~bvh();
      void bound(poly::Mesh* mesh);
      void metrics() const;
      bool hits(const poly::Ray &ray) const;
      void intersect(poly::Ray &ray, poly::Intersection& intersection) const;
      bvh_node* root;
      poly::Mesh* single_mesh;
   };
   
}


#endif //POLY_BVH_H
