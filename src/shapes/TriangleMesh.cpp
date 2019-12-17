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
         Polytope::Point vertex0 = Vertices[face.x];
         Polytope::Point vertex1 = Vertices[face.y];
         Polytope::Point vertex2 = Vertices[face.z];

         // step 1 - intersect with plane

         Polytope::Vector v0 = vertex0 - vertex1;
         Polytope::Vector v1 = vertex2 - vertex1;

         Polytope::Vector planeNormal = v0.Cross(v1);

         float t = -(objectSpaceRay.Origin.Dot(planeNormal)) - 
      }


      return false;
   }

   Point TriangleMesh::GetRandomPointOnSurface() {
      // TODO
      return Point();
   }
}
