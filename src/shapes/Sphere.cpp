//
// Created by Daniel on 20-Feb-18.
//

#include <cmath>
#include "Sphere.h"
#include "../Constants.h"

namespace Polytope {

   using Polytope::Transform;

   Sphere::Sphere(const Transform &ObjectToWorld) :
      AbstractShape(ObjectToWorld) { }

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

   Intersection Sphere::Intersect(const Ray &worldSpaceRay) {
      // TODO
      return Intersection();
   }


}
