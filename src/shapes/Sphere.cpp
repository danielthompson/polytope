//
// Created by Daniel on 20-Feb-18.
//

#include <cmath>
#include <utility>
#include <random>
#include "Sphere.h"
#include "../Constants.h"
#include "../structures/Vectors.h"

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

      objectSpaceRay = WorldToObject.Apply(worldSpaceRay);

      // TODO this can be simplified

      const Vector v = objectSpaceRay.Origin - Origin;

      const float a = objectSpaceRay.Direction.Dot(objectSpaceRay.Direction);
      const float b = 2 * (objectSpaceRay.Direction.Dot(v));
      const float c = (v).Dot(v) - (Radius * Radius);

      const float discriminant = (b * b) - (4 * a * c);

      if (discriminant < 0) {
         return false;
      }

      const float root = sqrt(discriminant);

      const float oneOverTwoA = .5f / a;

      const float t0 = (-b + root) * oneOverTwoA;

      const float t1 = (-b - root) * oneOverTwoA;

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
      if (ObjectToWorld.HasScale()) {
         Point objectSpaceIntersectionPoint = objectSpaceRay.GetPointAtT(hits);
         Point worldSpaceIntersectionPoint = ObjectToWorld.Apply(objectSpaceIntersectionPoint);
         hits = worldSpaceRay.GetTAtPoint(worldSpaceIntersectionPoint);
      }

      worldSpaceRay.MinT = hits < worldSpaceRay.MinT ? hits : worldSpaceRay.MinT;
      return true;
   }

   void Sphere::Intersect(Ray &worldSpaceRay, Intersection *intersection) {
      // TODO

      // we need to find the normal, for which we need the intersection point in object space

      Point worldSpaceIntersectionPoint = worldSpaceRay.GetPointAtT(worldSpaceRay.MinT);

      // spherical coordinate stuff



      intersection->Location = worldSpaceIntersectionPoint;

      Point objectSpaceIntersectionPoint = worldSpaceIntersectionPoint;

      //if (WorldToObject != null) {
         objectSpaceIntersectionPoint = WorldToObject.Apply(worldSpaceIntersectionPoint);
      //}

      // TODO this can be optimized

      float phi = std::atan2(objectSpaceIntersectionPoint.y, objectSpaceIntersectionPoint.x);
      float theta = std::acos(objectSpaceIntersectionPoint.z);
      float r = 1;

      float dxdu = -objectSpaceIntersectionPoint.y;
      float dydu = objectSpaceIntersectionPoint.x;
      float dzdu = 0.0f;

      Vector dpdu = Vector(dxdu, dydu, dzdu);

      float dxdv = objectSpaceIntersectionPoint.z * std::cos(phi);
      float dydv = objectSpaceIntersectionPoint.z * std::sin(phi);
      float dzdv = -Radius * std::sin(theta);

      Vector dpdv = Vector(dxdv, dydv, dzdv);

      // this can probably be deleted since Origin is always 0
      //objectSpaceIntersectionPoint -= Origin;

      intersection->Tangent1 = dpdu;

      Normal objectSpaceNormal = Normal(objectSpaceIntersectionPoint.x, objectSpaceIntersectionPoint.y, objectSpaceIntersectionPoint.z);

      Normal worldSpaceNormal = objectSpaceNormal;

      // transforms should never be null
      //if (ObjectToWorld != null) {
         worldSpaceNormal = ObjectToWorld.Apply(worldSpaceNormal);
      //}

      intersection->Normal = worldSpaceNormal;

      intersection->Tangent2 = dpdu.Cross(worldSpaceNormal);
   }

   Point Sphere::GetRandomPointOnSurface() {
      std::random_device rd{};
      std::mt19937 generator {rd()};

      std::normal_distribution<float> distribution{ 0.0f, 1.0f };

      distribution(generator);

      float x, y, z, d2;
      do {
         x = distribution(generator);
         y = distribution(generator);
         z = distribution(generator);
         d2 = x*x + y*y + z*z;
      } while (d2 <= DenormMin);
      float s = sqrt(1.0f / d2);
      return Point(x * s, y * s, z * s);
   }
}
