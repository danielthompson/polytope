//
// Created by Daniel on 15-Dec-19.
//

#include "TriangleMesh.h"

namespace Polytope {
   bool TriangleMesh::Hits(Ray &worldSpaceRay) const {
      return false;
   }

   void Polytope::TriangleMesh::Intersect(Polytope::Ray &worldSpaceRay, Polytope::Intersection *intersection) {
      Ray objectSpaceRay = WorldToObject->Apply(worldSpaceRay);

      //float minT = Infinity;

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

         if (t <= 0 || t >= worldSpaceRay.MinT)
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
            worldSpaceRay.MinT = t;
            intersection->faceIndex = faceIndex;
            intersection->Hits = true;
            intersection->Location = ObjectToWorld->Apply(hitPoint);

            // flip normal if needed
            Polytope::Normal n(planeNormal.x, planeNormal.y, planeNormal.z);
            if (objectSpaceRay.Direction.Dot(n) > 0) {
               n.Flip();
            }
            intersection->Normal = ObjectToWorld->Apply(n);
            intersection->Shape = this;

            const float edge0dot = std::abs(edge0.Dot(edge1));
            const float edge1dot = std::abs(edge1.Dot(edge2));
            const float edge2dot = std::abs(edge2.Dot(edge0));

            if (edge0dot > edge1dot && edge0dot > edge2dot) {
               intersection->Tangent1 = edge0;
               intersection->Tangent2 = edge1;
            }
            else if (edge1dot > edge0dot && edge1dot > edge2dot) {
               intersection->Tangent1 = edge1;
               intersection->Tangent2 = edge2;
            }
            else {
               intersection->Tangent1 = edge2;
               intersection->Tangent2 = edge0;
            }

            intersection->Tangent1.Normalize();
            intersection->Tangent2.Normalize();
         }
         faceIndex++;
      }
   }

   Point TriangleMesh::GetRandomPointOnSurface() {
      // TODO
      return Point();
   }
}
