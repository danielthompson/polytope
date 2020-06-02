//
// Created by Daniel on 20-Feb-18.
//

#include "Ray.h"
#include "Vectors.h"

namespace Polytope {

   using Polytope::Point;
   using Polytope::Vector;

   Ray::Ray(const Point &origin, const Vector &direction) :
         Origin(origin),
         Direction(direction),
         DirectionInverse(1.f / direction.x, 1.f / direction.y, 1.f / direction.z) { }

   Ray::Ray(const float ox, const float oy, const float oz, const float dx, const float dy, const float dz) :
      Origin(ox, oy, oz),
      Direction(dx, dy, dz),
      DirectionInverse(1.f / dx, 1.f / dy, 1.f / dz) { }

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

   Point Ray::GetPointAtT(const float t) const {
      return {
         std::fma(Direction.x, t, Origin.x),
         std::fma(Direction.y, t, Origin.y),
         std::fma(Direction.z, t, Origin.z),
      };
      //return Origin + (Direction * t);
   }

   void Ray::OffsetOriginForward(const float t) {
      Origin += (Direction * t);
   }

   void Ray::OffsetOrigin(const Normal &normal, const float t) {
      Origin.x = std::fma(normal.x, t, Origin.x);
      Origin.y = std::fma(normal.y, t, Origin.y);
      Origin.z = std::fma(normal.z, t, Origin.z);
//      Origin += normal * t;
   }
}
