//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractShape.h"
#include <utility>

namespace Polytope {

   //using Polytope::Transform;

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

   void BoundingBox::Intersect(Ray &worldSpaceRay, Intersection* intersection) const {
      float maxBoundFarT = FloatMax;
      float minBoundNearT = 0;

      const float gammaMultiplier = 1 + 2 * Gamma(3);
      
      intersection->Hits = true;

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
         intersection->Hits = false;
         return;
      }

      worldSpaceRay.MinT = minBoundNearT;
      //intersection.TMax = maxBoundFarT;

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
         intersection->Hits = false;
         return;
      }

      worldSpaceRay.MinT = minBoundNearT;
      //intersection.TMax = maxBoundFarT;

      // z
      tNear = (p0.z - worldSpaceRay.Origin.z) * worldSpaceRay.DirectionInverse.z;
      tFar = (p1.z - worldSpaceRay.Origin.z) * worldSpaceRay.DirectionInverse.z;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      if (minBoundNearT > maxBoundFarT) {
         intersection->Hits = false;
         return;
      }

      worldSpaceRay.MinT = minBoundNearT;
   }
}
