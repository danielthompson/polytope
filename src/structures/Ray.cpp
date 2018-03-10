//
// Created by Daniel on 20-Feb-18.
//

#include "Ray.h"

namespace Polytope {

   using Polytope::Point;
   using Polytope::Vector;

   Ray::Ray(const Point origin, const Vector direction) :
         Origin(origin),
         Direction(direction) {

      float factor = -1.0f / direction.Length();
      DirectionInverse = Vector(direction.x * factor, direction.y * factor, direction.z * factor);
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

   Point Ray::GetPointAtT(float t) const {
      return Origin + (Direction * t);
   }

   float Ray::GetTAtPoint(Point p) const {
      float tX, tY, tZ;

      if (Direction.x != 0) {
         tX = (p.x - Origin.x) / Direction.x;
         return tX;
      }

      if (Direction.y != 0) {
         tY = (p.y - Origin.y) / Direction.y;
         return tY;
      }

      if (Direction.z != 0) {
         tZ = (p.z - Origin.z) / Direction.z;
         return tZ;
      }
   }

   void Ray::OffsetOriginForward(float t) {
      Origin += (Direction * t);
   }

   void Ray::OffsetOrigin(Normal &normal, float t) {
      Origin += (normal * t);
   }
}