//
// Created by Daniel on 15-Dec-19.
//

#include <algorithm>
#include <vector>
#include "TriangleMesh.h"

namespace Polytope {

   namespace {
      float GetExtent(Polytope::Axis axis, const std::vector<Point3ui> &faces, const std::vector<Point> &vertices) {
         float min = Polytope::FloatMax;
         float max = -Polytope::FloatMax;

         for (const Point3ui &face : faces) {
            const float v0axisExtent = vertices[face.x][axis];
            const float v1axisExtent = vertices[face.y][axis];
            const float v2axisExtent = vertices[face.z][axis];

            if (v0axisExtent < min)
               min = v0axisExtent;
            if (v1axisExtent < min)
               min = v1axisExtent;
            if (v2axisExtent < min)
               min = v2axisExtent;

            if (v0axisExtent > max)
               max = v0axisExtent;
            if (v1axisExtent > max)
               max = v1axisExtent;
            if (v2axisExtent > max)
               max = v2axisExtent;
         }

         return max - min;
      }
   }

   bool TriangleMesh::Hits(Ray &worldSpaceRay) const {
      return false;
   }

   void
   TriangleMesh::IntersectFaces(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces) {
      unsigned int faceIndex = 0;

      for (const Point3ui &face : faces) {
         const Point vertex0 = Vertices[face.x];
         const Point vertex1 = Vertices[face.y];
         const Point vertex2 = Vertices[face.z];

         // step 1 - intersect with plane

         const Polytope::Vector edge0 = vertex1 - vertex0;
         const Polytope::Vector edge1 = vertex2 - vertex1;

         Polytope::Vector planeNormal = edge0.Cross(edge1);
         planeNormal.Normalize();

         const float divisor = planeNormal.Dot(ray.Direction);
         if (divisor == 0.0f) {
            // parallel
            continue;
         }

         const float t = planeNormal.Dot(vertex0 - ray.Origin) / divisor;

         if (t <= 0 || t >= ray.MinT)
            continue;

         const Polytope::Point hitPoint = ray.GetPointAtT(t);

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
            ray.MinT = t;
            intersection->faceIndex = faceIndex;
            intersection->Hits = true;
            intersection->Location = hitPoint;

            // flip normal if needed
            Polytope::Normal n(planeNormal.x, planeNormal.y, planeNormal.z);
            if (ray.Direction.Dot(n) > 0) {
               n.Flip();
            }
            intersection->Normal = n;
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

   void TriangleMesh::IntersectNode(Ray &ray, Intersection *intersection, BVHNode* node) {

      // base case
      if (node->low == nullptr) {
         IntersectFaces(ray, intersection, node->faces);
         return;
      }

      // recursive case
      if (node->low->bbox.Hits(ray)) {
         IntersectNode(ray, intersection, node->low);
      }
      if (root->high->bbox.Hits(ray)) {
         IntersectNode(ray, intersection, node->high);
      }
   }

   void Polytope::TriangleMesh::Intersect(Polytope::Ray &ray, Polytope::Intersection *intersection) {
      if (root != nullptr) {
         IntersectNode(ray, intersection, root);
      }
      else {
         IntersectFaces(ray, intersection, Faces);
      }
   }

   Point TriangleMesh::GetRandomPointOnSurface() {
      // TODO
      return Point();
   }

   void TriangleMesh::Split(Axis axis, float split) {
      switch (axis) {
         case Axis::x:
            SplitX(split);
            break;
         case Axis::y:
            SplitY(split);
            break;
         case Axis::z:
            SplitZ(split);
            break;
      }
   }

   void TriangleMesh::SplitX(const float x) {
      const Point pointOnPlane(x, 0, 0);
      const Normal normal(1, 0, 0);
      Split(pointOnPlane, normal);
   }

   void TriangleMesh::SplitY(const float y) {
      const Point pointOnPlane(0, y, 0);
      const Normal normal(0, 1, 0);
      Split(pointOnPlane, normal);
   }

   void TriangleMesh::SplitZ(const float z) {
      const Point pointOnPlane(0, 0, z);
      const Normal normal(0, 0, 1);
      Split(pointOnPlane, normal);
   }

   void TriangleMesh::Split(const Point &pointOnPlane, const Normal &normal) {
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

         // step 2 - skip if no split required

         if ((d0 >= 0 && d1 >= 0 && d2 >= 0)
         || (d0 <= 0 && d1 <= 0 && d2 <= 0))
            continue;

         // TODO - add optimization for the case in which the splitting plane happens to exactly
         // intersect an existing vertex, i.e.
         //
         //  |\
         //  | \
         //  |  \
         // -|--->---
         //  |  /
         //  | /
         //  |/
         // In such a case, there should be only 2 new faces, not three

         // step 3 - determine odd man out

         Point odd, even1, even2;
         unsigned int oddIndex, even1Index, even2Index;
         if ((d0 <= 0 && d1 >= 0 && d2 >= 0) || ((d0 >= 0 && d1 <= 0 && d2 <= 0))) {
            odd = v0;
            oddIndex = face.x;
            even1 = v1;
            even1Index = face.y;
            even2 = v2;
            even2Index = face.z;
         }
         else if ((d1 <= 0 && d2 >= 0 && d0 >= 0) || ((d1 >= 0 && d2 <= 0 && d0 <= 0))) {
            odd = v1;
            oddIndex = face.y;
            even1 = v2;
            even1Index = face.z;
            even2 = v0;
            even2Index = face.x;
         }
         else {
            odd = v2;
            oddIndex = face.z;
            even1 = v0;
            even1Index = face.x;
            even2 = v1;
            even2Index = face.y;
         }

         // step 4 - determine intersection points

         faceIndicesToDelete.push_back(i);
         const unsigned int vertexIndexStart = Vertices.size();
         const float numerator = (pointOnPlane - odd).Dot(normal);

         // calculate first intersection point
         {
            const Vector edge = even1 - odd;
            const float t = numerator / edge.Dot(normal);
            const Point intersectionPoint = odd + edge * t;
            Vertices.push_back(intersectionPoint);
         }

         // calculate second intersection point
         {
            const Vector edge = even2 - odd;
            const float t = numerator / edge.Dot(normal);
            const Point intersectionPoint = odd + edge * t;
            Vertices.push_back(intersectionPoint);
         }

         // step 5 - add new faces

         // add top face
         newFaces.emplace_back(oddIndex, vertexIndexStart, vertexIndexStart + 1);

         // add bottom faces
         newFaces.emplace_back(vertexIndexStart, even1Index, even2Index);
         newFaces.emplace_back(vertexIndexStart, even2Index, vertexIndexStart + 1);
      }

      if (!faceIndicesToDelete.empty()) {
         std::sort(faceIndicesToDelete.rbegin(), faceIndicesToDelete.rend());

         // delete old faces

         for (const int i : faceIndicesToDelete) {
            Faces.erase(Faces.begin() + i);
         }

         // add new faces
         for (const Point3ui &newFace : newFaces) {
            Faces.push_back(newFace);
         }
      }
   }

   void TriangleMesh::Bound() {
      if (root == nullptr)
         root = new BVHNode();

      const float extentX = GetExtent(Axis::x, Faces, Vertices);
      const float extentY = GetExtent(Axis::y, Faces, Vertices);
      const float extentZ = GetExtent(Axis::z, Faces, Vertices);

      Axis splitAxis = Axis::x;
      if (extentY > extentX && extentY > extentZ)
         splitAxis = Axis::y;
      else if (extentZ > extentX && extentZ > extentY)
         splitAxis = Axis::z;

      Bound(root, Faces, splitAxis);
   }

   void TriangleMesh::Bound(BVHNode *node, const std::vector<Point3ui> &faces, Axis axis) {

      // base case
      if (faces.size() < 10) {
         node->faces = faces;
         return;
      }

      // recursive case

      node->high = new BVHNode();
      node->low = new BVHNode();

      // determine split
      const float extent = GetExtent(axis, faces, Vertices);

      const float split = BoundingBox->p0[axis] + extent * 0.5f;

      // perform split
      Split(axis, split);

      for (const Point3ui &face : faces) {
         const Point v0 = Vertices[face.x];
         const Point v1 = Vertices[face.y];
         const Point v2 = Vertices[face.z];

         // add all faces at or below split to low child
         if (v0[axis] <= split && v1[axis] <= split && v2[axis] <= split) {
            node->low->faces.push_back(face);
         }
         else { // add all faces above split to high child
            node->high->faces.push_back(face);
         }
      }

      // set child bounding boxes
      node->low->bbox.p0 = BoundingBox->p0;
      node->low->bbox.p1 = BoundingBox->p1;

      node->high->bbox.p0 = BoundingBox->p0;
      node->high->bbox.p1 = BoundingBox->p1;

      switch (axis) {
         case Axis::x:
            node->low->bbox.p1.x = split;
            node->high->bbox.p0.x = split;
            break;
         case Axis::y:
            node->low->bbox.p1.y = split;
            node->high->bbox.p0.y = split;
            break;
         case Axis::z:
            node->low->bbox.p1.z = split;
            node->high->bbox.p0.z = split;
            break;
      }

      // figure out next splitting axis for children
      // low
      {
         const float extentX = GetExtent(Axis::x, node->low->faces, Vertices);
         const float extentY = GetExtent(Axis::y, node->low->faces, Vertices);
         const float extentZ = GetExtent(Axis::z, node->low->faces, Vertices);

         Axis splitAxis = Axis::x;
         if (extentY > extentX && extentY > extentZ)
            splitAxis = Axis::y;
         else if (extentZ > extentX && extentZ > extentY)
            splitAxis = Axis::z;

         // bug out if we're not actually making any progress
         if (node->low->faces.size() == node->faces.size() && axis == splitAxis)
            return;

         Bound(node->low, node->low->faces, splitAxis);
      }

      // high
      {
         const float extentX = GetExtent(Axis::x, node->high->faces, Vertices);
         const float extentY = GetExtent(Axis::y, node->high->faces, Vertices);
         const float extentZ = GetExtent(Axis::z, node->high->faces, Vertices);

         Axis splitAxis = Axis::x;
         if (extentY > extentX && extentY > extentZ)
            splitAxis = Axis::y;
         else if (extentZ > extentX && extentZ > extentY)
            splitAxis = Axis::z;

         // bug out if we're not actually making any progress
         if (node->high->faces.size() == node->faces.size() && axis == splitAxis)
            return;

         Bound(node->high, node->high->faces, splitAxis);
      }
   }
}
