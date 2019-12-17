//
// Created by Daniel on 15-Dec-19.
//

#include "TriangleMesh.h"

namespace Polytope {
   void Polytope::TriangleMesh::Intersect(const Polytope::Ray &worldSpaceRay, Polytope::Intersection *intersection) {
      // TODO
   }

   bool TriangleMesh::Hits(Ray &worldSpaceRay) const {
      Ray objectSpaceRay = WorldToObject.Apply(worldSpaceRay);

      for (const Point3ui &face : Faces) {
         const Polytope::Point vertex0 = Vertices[face.x];
         const Polytope::Point vertex1 = Vertices[face.y];
         const Polytope::Point vertex2 = Vertices[face.z];

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
