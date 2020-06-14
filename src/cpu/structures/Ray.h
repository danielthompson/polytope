//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLY_RAY_H
#define POLY_RAY_H

#include "Vectors.h"
#include "../constants.h"

namespace poly {

   class Ray {
   public:
      Point Origin;
      Vector Direction;
      //Vector DirectionInverse;
      float MinT = Infinity;
      float MaxT = Infinity;

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

#endif //POLY_RAY_H
