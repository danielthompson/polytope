//
// Created by Daniel Thompson on 12/23/19.
//

#include "BoundingBox.h"

namespace poly {
   bool BoundingBox::Hits(const poly::Ray &worldSpaceRay) const {
      float maxBoundFarT = poly::FloatMax;
      float minBoundNearT = 0;

      const float gammaMultiplier = 1 + 2 * poly::Gamma(3);

      // X
      float tNear = (p0.x - worldSpaceRay.Origin.x) * worldSpaceRay.DirectionInverse.x;
      float tFar = (p1.x - worldSpaceRay.Origin.x) * worldSpaceRay.DirectionInverse.x;

      float swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;
      if (minBoundNearT > maxBoundFarT) {
         return false;
      }

      // Y
      tNear = (p0.y - worldSpaceRay.Origin.y) * worldSpaceRay.DirectionInverse.y;
      tFar = (p1.y - worldSpaceRay.Origin.y) * worldSpaceRay.DirectionInverse.y;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      if (minBoundNearT > maxBoundFarT) {
         return false;
      }

      // z
      tNear = (p0.z - worldSpaceRay.Origin.z) * worldSpaceRay.DirectionInverse.z;
      tFar = (p1.z - worldSpaceRay.Origin.z) * worldSpaceRay.DirectionInverse.z;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      return minBoundNearT <= maxBoundFarT;

   }

   BoundingBox BoundingBox::Union(const BoundingBox &b) const {
      poly::Point min, max;
      min.x = p0.x < b.p0.x ? p0.x : b.p0.x;
      min.y = p0.y < b.p0.y ? p0.y : b.p0.y;
      min.z = p0.z < b.p0.z ? p0.z : b.p0.z;

      max.x = p1.x > b.p1.x ? p1.x : b.p1.x;
      max.y = p1.y > b.p1.y ? p1.y : b.p1.y;
      max.z = p1.z > b.p1.z ? p1.z : b.p1.z;

      return BoundingBox(min, max);
   }

   BoundingBox BoundingBox::Union(const poly::Point &p) const {
      poly::Point min = p0, max = p1;

      if (p.x < min.x)
         min.x = p.x;
      if (p.y < min.y)
         min.y = p.y;
      if (p.z < min.z)
         min.z = p.z;

      if (p.x > max.x)
         max.x = p.x;
      if (p.y > max.y)
         max.y = p.y;
      if (p.z > max.z)
         max.z = p.z;

      return BoundingBox(min, max);
   }

   void BoundingBox::UnionInPlace(const poly::Point &p) {
      if (p.x < p0.x)
         p0.x = p.x;
      if (p.y < p0.y)
         p0.y = p.y;
      if (p.z < p0.z)
         p0.z = p.z;

      if (p.x > p1.x)
         p1.x = p.x;
      if (p.y > p1.y)
         p1.y = p.y;
      if (p.z > p1.z)
         p1.z = p.z;
   }
}
