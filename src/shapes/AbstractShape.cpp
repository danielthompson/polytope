//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractShape.h"
#include <utility>

namespace Polytope {

   //using Polytope::Transform;

   bool BoundingBox::Hits(const Ray &worldSpaceRay) const {
      float maxBoundFarT = FloatMax;
      float minBoundNearT = 0;

      const float gammaMultiplier = 1 + 2 * Gamma(3);

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
      Polytope::Point min, max;
      min.x = p0.x < b.p0.x ? p0.x : b.p0.x;
      min.y = p0.y < b.p0.y ? p0.y : b.p0.y;
      min.z = p0.z < b.p0.z ? p0.z : b.p0.z;

      max.x = p1.x > b.p1.x ? p1.x : b.p1.x;
      max.y = p1.y > b.p1.y ? p1.y : b.p1.y;
      max.z = p1.z > b.p1.z ? p1.z : b.p1.z;

      return BoundingBox(min, max);
   }

   BoundingBox BoundingBox::Union(const Point &p) const {
      Polytope::Point min = p0, max = p1;

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

   void BoundingBox::UnionInPlace(const Point &p) {
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

   AbstractShape::AbstractShape(
         const Transform &objectToWorld,
         std::shared_ptr<Polytope::Material> material)
      : ObjectToWorld(objectToWorld),
        WorldToObject(objectToWorld.Invert()),
        Material(std::move(material)),
        BoundingBox(std::make_unique<Polytope::BoundingBox>()) { }

   AbstractShape::AbstractShape(
         const Transform &objectToWorld,
         const Transform &worldToObject,
         std::shared_ptr<Polytope::Material> material)
      : ObjectToWorld(objectToWorld),
        WorldToObject(worldToObject),
        Material(std::move(material)),
        BoundingBox(std::make_unique<Polytope::BoundingBox>()) {   }

   AbstractShape::AbstractShape(
         const Transform &objectToWorld,
         const Transform &worldToObject,
         std::shared_ptr<Polytope::Material> material,
         ShapeLight *light)
      : ObjectToWorld(objectToWorld),
        WorldToObject(worldToObject),
        Material(std::move(material)),
        Light(light),
        BoundingBox(std::make_unique<Polytope::BoundingBox>()) { };

}
