//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_RAY_H
#define POLYTOPE_RAY_H

#include "Point.h"
#include "Vector.h"

namespace Polytope {

   class Ray {
   public:
      Point Origin;
      Vector Direction;
      Vector DirectionInverse;
      float MinT;
      float MaxT;

      Ray(const Point &origin, const Vector &direction);

      bool operator==(const Ray &rhs) const;

      bool operator!=(const Ray &rhs) const;

   };

}

#endif //POLYTOPE_RAY_H
