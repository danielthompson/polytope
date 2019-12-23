//
// Created by Daniel Thompson on 12/23/19.
//

#ifndef POLYTOPE_BOUNDINGBOX_H
#define POLYTOPE_BOUNDINGBOX_H

#include "Vectors.h"
#include "Ray.h"

namespace Polytope {

   class BoundingBox {
   public:
      Polytope::Point p0, p1;

      BoundingBox() = default;
      BoundingBox(const Polytope::Point &min, const Polytope::Point &max) : p0(min), p1(max) {}

      bool Hits(const Polytope::Ray &worldSpaceRay) const;
      BoundingBox Union(const BoundingBox &b) const;
      BoundingBox Union(const Polytope::Point &p) const;
      void UnionInPlace(const Polytope::Point &p);
   };

}


#endif //POLYTOPE_BOUNDINGBOX_H
