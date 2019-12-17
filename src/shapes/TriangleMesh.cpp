//
// Created by Daniel on 15-Dec-19.
//

#include "TriangleMesh.h"

namespace Polytope {
   void Polytope::TriangleMesh::Intersect(const Polytope::Ray &worldSpaceRay, Polytope::Intersection *intersection) {
      Ray objectSpaceRay = WorldToObject.Apply(worldSpaceRay);

      float minT = Infinity;

      for (const Point3ui &face : Faces) {
         const Point vertex0 = Vertices[face.x];
         const Point vertex1 = Vertices[face.y];
         const Point vertex2 = Vertices[face.z];

         // step 1 - intersect with plane

         const Polytope::Vector v0 = vertex0 - vertex1;
         const Polytope::Vector v1 = vertex1 - vertex2;

         const Polytope::Vector planeNormal = v0.Cross(v1);

         const float divisor = planeNormal.Dot(objectSpaceRay.Direction);
         if (divisor == 0.0f) {
            // parallel
            continue;
         }

         const float t = planeNormal.Dot(vertex0 - objectSpaceRay.Origin) / divisor;

         if (t < minT && t > 0) {
            minT = t;
            intersection->Normal = Polytope::Normal(planeNormal.x, planeNormal.y, planeNormal.z);
         }
      }

      if (minT != Infinity) {
         intersection->Hits = true;
         intersection->Shape = this;
         Point worldSpacePoint = worldSpaceRay.GetPointAtT(minT);
         intersection->Location = worldSpacePoint;

         // flip normal if needed
         if (objectSpaceRay.Direction.Dot(intersection->Normal) > 0)
            intersection->Normal.Flip();
      }
   }

   bool TriangleMesh::Hits(Ray &worldSpaceRay) const {
      Ray objectSpaceRay = WorldToObject.Apply(worldSpaceRay);

      for (const Point3ui &face : Faces) {
         const Point vertex0 = Vertices[face.x];
         const Point vertex1 = Vertices[face.y];
         const Point vertex2 = Vertices[face.z];

         // step 1 - intersect with plane

         const Polytope::Vector v0 = vertex0 - vertex1;
         const Polytope::Vector v1 = vertex1 - vertex2;

         const Polytope::Vector planeNormal = v0.Cross(v1);

         const float divisor = planeNormal.Dot(objectSpaceRay.Direction);
         if (divisor == 0.0f) {
            // parallel
            continue;
         }

         const float t = planeNormal.Dot(vertex0 - objectSpaceRay.Origin) / divisor;

         if (t > 0)
            return true;
      }

      return false;
   }

   Point TriangleMesh::GetRandomPointOnSurface() {
      // TODO
      return Point();
   }
}
