//
// Created by Daniel on 20-Feb-18.
//

#include <cmath>
#include <utility>
#include "Sphere.h"
#include "../Constants.h"
#include "../structures/Normal.h"

namespace Polytope {

   const Point Sphere::Origin = Point(0, 0, 0);

   using Polytope::Transform;

   Sphere::Sphere(const Transform &objectToWorld, std::shared_ptr<Polytope::Material> material) : AbstractShape(
         objectToWorld, std::move(material)) {  }

   Sphere::Sphere(const Transform &objectToWorld, const Transform &worldToObject,
                  std::shared_ptr<Polytope::Material> material) : AbstractShape(objectToWorld, worldToObject,
                                                                                std::move(material)) {  }

   bool Sphere::Hits(Ray &worldSpaceRay) const {
      Ray objectSpaceRay = worldSpaceRay;

      //if (WorldToObject != 0) {
      objectSpaceRay = WorldToObject.Apply(worldSpaceRay);
      //}


      // TODO this can be simplified

      float a = objectSpaceRay.Direction.Dot(objectSpaceRay.Direction);
      float b = 2 * (objectSpaceRay.Direction.Dot(objectSpaceRay.Origin - Origin));
      float c = (objectSpaceRay.Origin - Origin).Dot(objectSpaceRay.Origin - Origin) - (Radius * Radius);

      float discriminant = (b * b) - (4 * a * c);

      if (discriminant < 0) {
         return false;
      }

      float root = (float) sqrt(discriminant);

      float oneOverTwoA = .5f / a;

      float t0 = (-b + root) * oneOverTwoA;

      float t1 = (-b - root) * oneOverTwoA;

      float hits;

      if (t1 < Epsilon) {
         hits = (t0 >= Epsilon) ? t0 : NOHIT;
      }
      else if (WithinEpsilon(t1, 0)) {
         hits = t0 < Epsilon ? t1 : t0;
      }
      else {
         if (t0 < Epsilon) {
            hits = t1;
         }
         else if (WithinEpsilon(t0, 0)) {
            hits = t0;
         }
         else {
            hits = t0 < t1 ? t0 : t1;
         }
      }

      if (hits == NOHIT)
         return false;

      if (hits < Epsilon)
         return false;

      // convert T back to world space
      if (/*ObjectToWorld != null && */ObjectToWorld.HasScale()) {
         Point objectSpaceIntersectionPoint = objectSpaceRay.GetPointAtT(hits);
         Point worldSpaceIntersectionPoint = ObjectToWorld.Apply(objectSpaceIntersectionPoint);
         hits = worldSpaceRay.GetTAtPoint(worldSpaceIntersectionPoint);
      }

      worldSpaceRay.MinT = hits < worldSpaceRay.MinT ? hits : worldSpaceRay.MinT;
      return true;
   }

   void Sphere::Intersect(const Ray &worldSpaceRay, Intersection *intersection) {
      // TODO

      // we need to find the normal, for which we need the intersectionpoint in object space

      Point worldSpaceIntersectionPoint = worldSpaceRay.GetPointAtT(worldSpaceRay.MinT);



      intersection->Location = worldSpaceIntersectionPoint;

      Point objectSpaceIntersectionPoint = worldSpaceIntersectionPoint;

      //if (WorldToObject != null) {
         objectSpaceIntersectionPoint = WorldToObject.Apply(worldSpaceIntersectionPoint);
      //}

      // this can probably be deleted since Origin is always 0
      //objectSpaceIntersectionPoint -= Origin;

      Normal objectSpaceNormal = Normal(objectSpaceIntersectionPoint.x, objectSpaceIntersectionPoint.y, objectSpaceIntersectionPoint.z);

      Normal worldSpaceNormal = objectSpaceNormal;

      // transforms should never be null
      //if (ObjectToWorld != null) {
         worldSpaceNormal = ObjectToWorld.Apply(worldSpaceNormal);
      //}

      worldSpaceNormal.Normalize();

      intersection->Normal = worldSpaceNormal;
   }

   Point Sphere::GetRandomPointOnSurface() const {
      return Point();
   }


}
