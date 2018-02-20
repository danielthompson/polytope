//
// Created by Daniel on 20-Feb-18.
//

#include "Ray.h"

namespace Polytope {

   using Polytope::Point;
   using Polytope::Vector;

   Ray::Ray(const Point &origin, const Vector &direction) :
         Origin(origin),
         Direction(direction),
         DirectionInverse(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z) {

   }

   bool Ray::operator==(const Ray &rhs) const {
      return Origin == rhs.Origin &&
             Direction == rhs.Direction &&
             DirectionInverse == rhs.DirectionInverse &&
             MinT == rhs.MinT &&
             MaxT == rhs.MaxT;
   }

   bool Ray::operator!=(const Ray &rhs) const {
      return !(rhs == *this);
   }
}