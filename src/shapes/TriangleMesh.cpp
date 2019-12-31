//
// Created by Daniel on 15-Dec-19.
//

#include <algorithm>
#include <vector>
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
         const Point vertex0 = Vertices[face.x];
         const Point vertex1 = Vertices[face.y];
         const Point vertex2 = Vertices[face.z];

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



   void TriangleMesh::SplitY(const float y) {
      const Point pointOnPlane(0, y, 0);
      const Normal normal(0, 1, 0);

      std::vector<Point3ui> newFaces;
      std::vector<int> faceIndicesToDelete;

      for (unsigned int i = 0; i < Faces.size(); i++) {

         const Point3ui face = Faces[i];
         const Point v0 = Vertices[face.x];
         const Point v1 = Vertices[face.y];
         const Point v2 = Vertices[face.z];

         // step 1 - determine the signed distance to the plane for all 3 points

         const float d0 = Polytope::SignedDistanceFromPlane(pointOnPlane, normal, v0);
         const float d1 = Polytope::SignedDistanceFromPlane(pointOnPlane, normal, v1);
         const float d2 = Polytope::SignedDistanceFromPlane(pointOnPlane, normal, v2);

         // step 2 - determine if any faces need to be split

         // if d0 is the odd man out
         if ((d0 < 0 && d1 > 0 && d2 > 0) || ((d0 > 0 && d1 < 0 && d2 < 0))) {
            faceIndicesToDelete.push_back(i);
            const unsigned int vertexIndexStart = Vertices.size() - 1;

            // calculate first intersection point
            const Vector e0 = v1 - v0;
            const float t0 = (y - v0.y) / e0.y;
            const Point i0 = v0 + e0 * t0;
            Vertices.push_back(i0);

            // calculate second intersection point
            const Vector e1 = v2 - v0;
            const float t1 = (y - v0.y) / e1.y;
            const Point i1 = v0 + e1 * t1;
            Vertices.push_back(i1);

            // add top face
            newFaces.emplace_back(face.x, vertexIndexStart + 1, vertexIndexStart + 2);

            // add bottom faces
            newFaces.emplace_back(vertexIndexStart + 1, face.y, face.z);
            newFaces.emplace_back(vertexIndexStart + 1, face.z, vertexIndexStart + 2);
         }
      }

      if (!faceIndicesToDelete.empty()) {
         std::sort(faceIndicesToDelete.begin(), faceIndicesToDelete.end());

         // delete old faces

         for (auto i = faceIndicesToDelete.size(); i > 0; i--) {
            Faces.erase(Faces.begin() + i - 1);
         }

         // add new faces
         for (const Point3ui &newFace : newFaces) {
            Faces.push_back(newFace);
         }
      }
   }
}
