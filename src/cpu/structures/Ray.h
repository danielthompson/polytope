//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_RAY_H
#define POLYTOPE_RAY_H

#include "Vectors.h"
#include "../constants.h"

namespace Polytope {

   class Ray {
   public:
      Point Origin;
      Vector Direction;
      Vector DirectionInverse;
      float MinT = Infinity;
      float MaxT = Infinity;

      int x, y;

      Ray() : MinT(Infinity), MaxT(Infinity) { };
      Ray(const Point &origin, const Vector &direction);
      Ray(float ox, float oy, float oz, float dx, float dy, float dz);

      bool operator==(const Ray &rhs) const;
      bool operator!=(const Ray &rhs) const;

      Point GetPointAtT(float t) const;
      void OffsetOriginForward(float t);
      void OffsetOrigin(const Normal &normal, float t);
   };
}

#endif //POLYTOPE_RAY_H
