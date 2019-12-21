//
// Created by Daniel on 15-Dec-19.
//

#include "TriangleMesh.h"

namespace Polytope {
   bool TriangleMesh::Hits(Ray &worldSpaceRay) const {
      Ray objectSpaceRay = WorldToObject.Apply(worldSpaceRay);

      float minT = Infinity;

      unsigned int faceIndex = 0;

      for (const Point3ui &face : Faces) {
         const Point vertex0 = Vertices[face.x - 1];
         const Point vertex1 = Vertices[face.y - 1];
         const Point vertex2 = Vertices[face.z - 1];

         // step 1 - intersect with plane

         const Polytope::Vector edge0 = vertex1 - vertex0;
         const Polytope::Vector edge1 = vertex2 - vertex1;

         Polytope::Vector planeNormal = edge0.Cross(edge1);
         planeNormal.Normalize();

         const float divisor = planeNormal.Dot(objectSpaceRay.Direction);
         if (divisor == 0.0f) {
            // parallel
            continue;
         }

         const float t = planeNormal.Dot(vertex0 - objectSpaceRay.Origin) / divisor;

         if (t <= 0 || t >= minT)
            continue;

         const Polytope::Point hitPoint = objectSpaceRay.GetPointAtT(t);

         // step 2 - inside/outside test

         const Polytope::Vector edge2 = vertex0 - vertex2;

         // 1. Compute the cross product of [the vector defined by the two edges' vertices] and [the vector defined by the first edge's vertex and the point]
         const Polytope::Vector p0 = hitPoint - vertex0;
         const Polytope::Vector cross0 = edge0.Cross(p0);

         // 2. Compute the dot product of the result from [1] and the polygon's normal
         const float normal0 = cross0.Dot(planeNormal);

         // 3. The sign of [2] determines if the point is on the right or left side of that edge.
         const bool pos0 = normal0 > 0;

         // 4. Repeat for all 3 edges
         const Polytope::Vector p1 = hitPoint - vertex1;
         const Polytope::Vector cross1 = edge1.Cross(p1);
         const float normal1 = cross1.Dot(planeNormal);
         const bool pos1 = normal1 > 0;

         const Polytope::Vector p2 = hitPoint - vertex2;
         const Polytope::Vector cross2 = edge2.Cross(p2);
         const float normal2 = cross2.Dot(planeNormal);
         const bool pos2 = normal2 > 0;

         bool inside = pos0 && pos1 && pos2;

         if (inside) {
            minT = t;
         }
         faceIndex++;
      }

      if (minT == Infinity)
         return false;

      worldSpaceRay.MinT = minT < worldSpaceRay.MinT ? minT : worldSpaceRay.MinT;

      return true;
   }

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

         Polytope::Vector planeNormal = v0.Cross(v1);
         planeNormal.Normalize();

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



   Point TriangleMesh::GetRandomPointOnSurface() {
      // TODO
      return Point();
   }
}
