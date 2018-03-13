//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_RAY_H
#define POLYTOPE_RAY_H

#include "Point.h"
#include "Vector.h"
#include "../Constants.h"

namespace Polytope {

   class Ray {
   public:

      // methods

      Ray() : MinT(Infinity), MaxT(Infinity) { };
      Ray(const Point &origin, const Vector &direction);

      bool operator==(const Ray &rhs) const;
      bool operator!=(const Ray &rhs) const;

      Point GetPointAtT(float t) const;
      float GetTAtPoint(const Point &p) const;
      void OffsetOriginForward(float t);
      void OffsetOrigin(Normal &normal, float t);

      // data

      Point Origin;
      Vector Direction;
      Vector DirectionInverse;
      float MinT = Infinity;
      float MaxT = Infinity;
   };

}

#endif //POLYTOPE_RAY_H
