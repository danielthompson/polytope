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

      Ray() : MaxT(Infinity) { };
      Ray(const Point &origin, const Vector &direction);

      bool operator==(const Ray &rhs) const;
      bool operator!=(const Ray &rhs) const;

      Point GetPointAtT(float t) const;
      float GetTAtPoint(Point p) const;

      // data

      Point Origin;
      Vector Direction;
      Vector DirectionInverse;
      float MinT;
      float MaxT;
   };

}

#endif //POLYTOPE_RAY_H
