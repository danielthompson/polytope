//
// Created by Daniel Thompson on 12/23/19.
//

#ifndef POLY_BOUNDINGBOX_H
#define POLY_BOUNDINGBOX_H

#include "Vectors.h"
#include "ray.h"

namespace poly {

   class bounding_box {
   public:
      poly::point p0, p1;

      bounding_box() = default;
      bounding_box(const poly::point &min, const poly::point &max) : p0(min), p1(max) {}

      bool hits(const poly::ray &world_space_ray, const poly::vector& inverse_direction) const;
      poly::bounding_box Union(const poly::bounding_box &b) const;
      poly::bounding_box Union(const poly::point &p) const;
      void union_in_place(const poly::point &p);
      void union_in_place(const poly::bounding_box &b);
      
      float surface_area() const;
   };

}


#endif //POLY_BOUNDINGBOX_H
