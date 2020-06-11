//
// Created by daniel on 6/9/20.
//

#ifndef POLY_BVH_H
#define POLY_BVH_H

#include "../structures/BoundingBox.h"
#include "../shapes/linear_soa/mesh_linear_soa.h"
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
      void bound(poly::TMesh* mesh);
      void metrics() const;
      bool hits(const poly::Ray &ray);
      bvh_node* root;
   };
   
}


#endif //POLY_BVH_H
