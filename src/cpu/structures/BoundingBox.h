//
// Created by Daniel Thompson on 12/23/19.
//

#ifndef POLY_BOUNDINGBOX_H
#define POLY_BOUNDINGBOX_H

#include "Vectors.h"
#include "Ray.h"

namespace poly {

   class BoundingBox {
   public:
      poly::Point p0, p1;

      BoundingBox() = default;
      BoundingBox(const poly::Point &min, const poly::Point &max) : p0(min), p1(max) {}

      bool Hits(const poly::Ray &worldSpaceRay, const poly::Vector& inverse_direction) const;
      BoundingBox Union(const BoundingBox &b) const;
      BoundingBox Union(const poly::Point &p) const;
      void UnionInPlace(const poly::Point &p);
   };

}


#endif //POLY_BOUNDINGBOX_H
