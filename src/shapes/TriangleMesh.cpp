//
// Created by Daniel on 15-Dec-19.
//

#include <algorithm>
#include <vector>
#include <map>
#include <queue>
#include "TriangleMesh.h"
#include "../utilities/Common.h"
#include "triangle_mesh_ispc.h"

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

         const Polytope::Vector p0 = hitPoint - vertex0;
         const Polytope::Vector cross0 = edge0.Cross(p0);
         const float normal0 = cross0.Dot(planeNormal);
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
            intersection->Hits = true;
            ray.MinT = t;
            intersection->faceIndex = faceIndex;
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

   void TriangleMesh::IntersectFacesISPC(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces) {

      //unsigned int faceIndex = 0;

      float ispc_vertex[9] = { 0 };
      float ispc_edge[9] = { 0 };
      bool ispc_out[3] = { false };

      for (const Point3ui &face : faces) {
         const Point vertex0 = Vertices[face.x];
         const Point vertex1 = Vertices[face.y];
         const Point vertex2 = Vertices[face.z];

         ispc_vertex[0] = vertex0.x;
         ispc_vertex[1] = vertex0.y;
         ispc_vertex[2] = vertex0.z;
         ispc_vertex[3] = vertex1.x;
         ispc_vertex[4] = vertex1.y;
         ispc_vertex[5] = vertex1.z;
         ispc_vertex[6] = vertex2.x;
         ispc_vertex[7] = vertex2.y;
         ispc_vertex[8] = vertex2.z;

         ispc_out[0] = false;
         ispc_out[1] = false;
         ispc_out[2] = false;

         // step 1 - intersect with plane

         // const Polytope::Vector edge0 = vertex1 - vertex0;
         ispc_edge[0] = ispc_vertex[3] - ispc_vertex[0];
         ispc_edge[1] = ispc_vertex[4] - ispc_vertex[1];
         ispc_edge[2] = ispc_vertex[5] - ispc_vertex[2];

         // const Polytope::Vector edge1 = vertex2 - vertex1;
         ispc_edge[3] = ispc_vertex[6] - ispc_vertex[3];
         ispc_edge[4] = ispc_vertex[7] - ispc_vertex[4];
         ispc_edge[5] = ispc_vertex[8] - ispc_vertex[5];

         //Polytope::Vector planeNormal = edge0.Cross(edge1);
         Polytope::Vector planeNormal = Polytope::Vector(
               ispc_edge[1] * ispc_edge[5] - ispc_edge[2] * ispc_edge[4],
               ispc_edge[2] * ispc_edge[3] - ispc_edge[0] * ispc_edge[5],
               ispc_edge[0] * ispc_edge[4] - ispc_edge[1] * ispc_edge[3]
         );
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

         // const Polytope::Vector edge2 = vertex0 - vertex2;
         ispc_edge[6] = ispc_vertex[0] - ispc_vertex[6];
         ispc_edge[7] = ispc_vertex[1] - ispc_vertex[7];
         ispc_edge[8] = ispc_vertex[2] - ispc_vertex[8];

         ispc::simple(hitPoint.x, hitPoint.y, hitPoint.z, planeNormal.x, planeNormal.y, planeNormal.z, ispc_vertex, ispc_edge, ispc_out, 3);

         bool inside = ispc_out[0] && ispc_out[1] && ispc_out[2];

         if (inside) {
            intersection->Hits = true;
            ray.MinT = t;
            //intersection->faceIndex = faceIndex;
            intersection->Location = hitPoint;

            // flip normal if needed
            Polytope::Normal n(planeNormal.x, planeNormal.y, planeNormal.z);
            if (ray.Direction.Dot(n) > 0) {
               n.Flip();
            }

            intersection->Normal = n;
            intersection->Shape = this;

            const float edge0dot = std::abs(ispc_edge[0] * ispc_edge[3] + ispc_edge[1] + ispc_edge[4] + ispc_edge[2] * ispc_edge[5]);
            const float edge1dot = std::abs(ispc_edge[3] * ispc_edge[6] + ispc_edge[4] + ispc_edge[7] + ispc_edge[5] * ispc_edge[8]);
            const float edge2dot = std::abs(ispc_edge[6] * ispc_edge[0] + ispc_edge[7] + ispc_edge[1] + ispc_edge[8] * ispc_edge[2]);

            if (edge0dot > edge1dot && edge0dot > edge2dot) {
               intersection->Tangent1 = Polytope::Vector(ispc_edge[0], ispc_edge[1], ispc_edge[2]);
               intersection->Tangent2 = Polytope::Vector(ispc_edge[3], ispc_edge[4], ispc_edge[5]);
            }
            else if (edge1dot > edge0dot && edge1dot > edge2dot) {
               intersection->Tangent1 = Polytope::Vector(ispc_edge[3], ispc_edge[4], ispc_edge[5]);
               intersection->Tangent2 = Polytope::Vector(ispc_edge[6], ispc_edge[7], ispc_edge[8]);
            }
            else {
               intersection->Tangent1 = Polytope::Vector(ispc_edge[6], ispc_edge[7], ispc_edge[8]);
               intersection->Tangent2 = Polytope::Vector(ispc_edge[0], ispc_edge[1], ispc_edge[2]);;
            }

            intersection->Tangent1.Normalize();
            intersection->Tangent2.Normalize();
         }
         //faceIndex++;
      }
   }


   void TriangleMesh::IntersectFacesISPC_SOA(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces) {

      //unsigned int faceIndex = 0;

      float ispc_vertex[9] = { 0 };
      float ispc_edge[9] = { 0 };
      bool ispc_out[3] = { false };

      for (const Point3ui &face : faces) {
         const Point vertex0 = Vertices[face.x];
         const Point vertex1 = Vertices[face.y];
         const Point vertex2 = Vertices[face.z];

         ispc_vertex[0] = vertex0.x;
         ispc_vertex[1] = vertex1.x;
         ispc_vertex[2] = vertex2.x;
         ispc_vertex[3] = vertex0.y;
         ispc_vertex[4] = vertex1.y;
         ispc_vertex[5] = vertex2.y;
         ispc_vertex[6] = vertex0.z;
         ispc_vertex[7] = vertex1.z;
         ispc_vertex[8] = vertex2.z;

         ispc_out[0] = false;
         ispc_out[1] = false;
         ispc_out[2] = false;

         // step 1 - intersect with plane

         // const Polytope::Vector edge0 = vertex1 - vertex0;
         ispc_edge[0] = ispc_vertex[1] - ispc_vertex[0]; //e0x
         ispc_edge[3] = ispc_vertex[4] - ispc_vertex[3]; //e0y -> 3
         ispc_edge[6] = ispc_vertex[7] - ispc_vertex[6]; //e0z -> 6

         // const Polytope::Vector edge1 = vertex2 - vertex1;
         ispc_edge[1] = ispc_vertex[2] - ispc_vertex[1]; //e1x -> 1
         ispc_edge[4] = ispc_vertex[5] - ispc_vertex[4]; //e1y -> 4
         ispc_edge[7] = ispc_vertex[8] - ispc_vertex[7]; //e1z -> 7

         //Polytope::Vector planeNormal = edge0.Cross(edge1);
         Polytope::Vector planeNormal = Polytope::Vector(
               ispc_edge[3] * ispc_edge[7] - ispc_edge[6] * ispc_edge[4],
               ispc_edge[6] * ispc_edge[1] - ispc_edge[0] * ispc_edge[7],
               ispc_edge[0] * ispc_edge[4] - ispc_edge[3] * ispc_edge[1]
         );
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

         // const Polytope::Vector edge2 = vertex0 - vertex2;
         ispc_edge[2] = ispc_vertex[0] - ispc_vertex[2]; // e2x -> 2
         ispc_edge[5] = ispc_vertex[3] - ispc_vertex[5]; // e2y -> 5
         ispc_edge[8] = ispc_vertex[6] - ispc_vertex[8]; // e2z -> 8

         ispc::simple(hitPoint.x, hitPoint.y, hitPoint.z, planeNormal.x, planeNormal.y, planeNormal.z, ispc_vertex, ispc_edge, ispc_out, 3);

         bool inside = ispc_out[0] && ispc_out[1] && ispc_out[2];

         if (inside) {
            intersection->Hits = true;
            ray.MinT = t;
            //intersection->faceIndex = faceIndex;
            intersection->Location = hitPoint;

            // flip normal if needed
            Polytope::Normal n(planeNormal.x, planeNormal.y, planeNormal.z);
            if (ray.Direction.Dot(n) > 0) {
               n.Flip();
            }

            intersection->Normal = n;
            intersection->Shape = this;

            const float edge0dot = std::abs(ispc_edge[0] * ispc_edge[1] + ispc_edge[3] + ispc_edge[4] + ispc_edge[6] * ispc_edge[7]);
            const float edge1dot = std::abs(ispc_edge[1] * ispc_edge[2] + ispc_edge[4] + ispc_edge[5] + ispc_edge[7] * ispc_edge[8]);
            const float edge2dot = std::abs(ispc_edge[2] * ispc_edge[0] + ispc_edge[5] + ispc_edge[3] + ispc_edge[8] * ispc_edge[6]);

            if (edge0dot > edge1dot && edge0dot > edge2dot) {
               intersection->Tangent1 = Polytope::Vector(ispc_edge[0], ispc_edge[3], ispc_edge[6]);
               intersection->Tangent2 = Polytope::Vector(ispc_edge[1], ispc_edge[4], ispc_edge[7]);
            }
            else if (edge1dot > edge0dot && edge1dot > edge2dot) {
               intersection->Tangent1 = Polytope::Vector(ispc_edge[1], ispc_edge[4], ispc_edge[7]);
               intersection->Tangent2 = Polytope::Vector(ispc_edge[2], ispc_edge[5], ispc_edge[8]);
            }
            else {
               intersection->Tangent1 = Polytope::Vector(ispc_edge[2], ispc_edge[5], ispc_edge[8]);
               intersection->Tangent2 = Polytope::Vector(ispc_edge[0], ispc_edge[3], ispc_edge[6]);;
            }

            intersection->Tangent1.Normalize();
            intersection->Tangent2.Normalize();
         }
         //faceIndex++;
      }
   }

   void TriangleMesh::IntersectNode(Ray &ray, Intersection *intersection, BVHNode* node, const unsigned int depth) {

      bool debug = false;

      // base case
      if (node->low == nullptr) {
         if (ray.x == 130 && ray.y == 128)
            debug = true;
         IntersectFacesISPC_SOA(ray, intersection, node->faces);
         //IntersectFaces(ray, intersection, node->faces);

         return;
      }

      if (ray.x == 130 && ray.y == 128)
         debug = true;

      // TODO avoid looking in far node if there's a hit in near node
      // it ought to be possible to figure out which side of the split the ray's origin is on,
      // which means the child bounding box on that same side is the nearer one.
      // since we use mesh splitting, sibling boxes never overlap.
      // => therefore, if there's a hit in the near one, a hit in the far one can't be closer.
      // so we can avoid looking in the far one altogether.

      // figure out near node
      // check that one first

      // recursive case

      bool lowHits = node->low->bbox.Hits(ray);
      bool highHits = node->high->bbox.Hits(ray);

      if (!lowHits && !highHits && ray.x == 130 && ray.y == 128)
         debug = true;

      if (lowHits) {
         IntersectNode(ray, intersection, node->low, depth + 1);
      }
      if (highHits) {
         IntersectNode(ray, intersection, node->high, depth + 1);
      }
   }

   void Polytope::TriangleMesh::Intersect(Polytope::Ray &ray, Polytope::Intersection *intersection) {
      if (root != nullptr) {
         IntersectNode(ray, intersection, root, 0);
      }
      else {
         IntersectFaces(ray, intersection, Faces);
      }

      if (ray.x == 130 && ray.y == 128) {
         bool debug = true;
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
         std::sort(faceIndicesToDelete.begin(), faceIndicesToDelete.end());

         // delete old faces

         std::vector<Point3ui> newFaceVector;
         newFaceVector.reserve(Faces.size() - faceIndicesToDelete.size() + newFaces.size());

         unsigned int faceIndicesToDeleteIndex = 0;

         for (unsigned int i = 0; i < Faces.size(); i++) {
            if (i != faceIndicesToDelete[faceIndicesToDeleteIndex]) {
               newFaceVector.push_back(Faces[i]);
            }
            else {
               faceIndicesToDeleteIndex++;
            }
         }

         Faces = newFaceVector;

         // add new faces
         for (const Point3ui &newFace : newFaces) {
            Faces.push_back(newFace);
         }
      }
   }

   void TriangleMesh::Bound() {
      if (root == nullptr)
         root = new BVHNode();

      Bound(root, Faces, 0);
   }

   void TriangleMesh:: Bound(BVHNode *node, const std::vector<Point3ui> &faces, const unsigned int depth) {

      node->ShrinkBoundingBox(Vertices, faces);

      // base case
      if (faces.size() < 750 || depth > 10) {
         node->faces = faces;
         return;
      }

      // recursive case

      // decide on split axis

      // TODO - instead of just picking the widest extent and halving it,
      // pick the axis with the most unique vertices in that axis, and split on the median.

      const float extentX = GetExtent(Axis::x, faces, Vertices);
      const float extentY = GetExtent(Axis::y, faces, Vertices);
      const float extentZ = GetExtent(Axis::z, faces, Vertices);

      float extent = extentX;
      Axis splitAxis = Axis::x;
      if (extentY > extentX && extentY > extentZ) {
         splitAxis = Axis::y;
         extent = extentY;
      }
      else if (extentZ > extentX && extentZ > extentY) {
         splitAxis = Axis::z;
         extent = extentZ;
      }
      const float split = node->bbox.p0[splitAxis] + extent * 0.5f;

      // perform split
      Split(splitAxis, split);

      // put faces in high or low
      std::vector<Point3ui> lowFaces, highFaces;

      for (const Point3ui &face : faces) {
         const Point v0 = Vertices[face.x];
         const Point v1 = Vertices[face.y];
         const Point v2 = Vertices[face.z];

         // add all faces at or below split to low child
         if (v0[splitAxis] <= split && v1[splitAxis] <= split && v2[splitAxis] <= split) {
            lowFaces.push_back(face);
         }
         else { // add all faces above split to high child
            highFaces.push_back(face);
         }
      }

      // check to see whether we came up with a good split and should continue

      bool continueRecursing = true;
      if (faces.size() <= lowFaces.size()
      || faces.size() <= highFaces.size()
      || lowFaces.empty()
      || highFaces.empty())
         continueRecursing = false;

      if (continueRecursing) {
         node->high = new BVHNode();
         node->low = new BVHNode();
         node->high->parent = node;
         node->low->parent = node;

         // set child bounding boxes
         node->low->bbox.p0 = BoundingBox->p0;
         node->low->bbox.p1 = BoundingBox->p1;

         node->high->bbox.p0 = BoundingBox->p0;
         node->high->bbox.p1 = BoundingBox->p1;

         Bound(node->low, lowFaces, depth + 1);
         Bound(node->high, highFaces, depth + 1);
      }
      else {
         node->faces = faces;
      }
   }

   void TriangleMesh::CalculateVertexNormals() {

      Normals = std::vector<Normal>(Vertices.size(), Normal(0, 0, 0));

      for (const Point3ui &face : Faces) {
         const Point vertex0 = Vertices[face.x];
         const Point vertex1 = Vertices[face.y];
         const Point vertex2 = Vertices[face.z];

         // step 1 - intersect with plane

         const Polytope::Vector edge0 = vertex1 - vertex0;
         const Polytope::Vector edge1 = vertex2 - vertex1;

         Polytope::Vector planeNormal = edge0.Cross(edge1);
         planeNormal.Normalize();

         Normals[face.x].x += planeNormal.x;
         Normals[face.x].y += planeNormal.y;
         Normals[face.x].z += planeNormal.z;

         Normals[face.y].x += planeNormal.x;
         Normals[face.y].y += planeNormal.y;
         Normals[face.y].z += planeNormal.z;

         Normals[face.z].x += planeNormal.x;
         Normals[face.z].y += planeNormal.y;
         Normals[face.z].z += planeNormal.z;
      }

      for (Normal &normal : Normals) {
         normal.Normalize();
      }
   }

   unsigned int TriangleMesh::CountUniqueVertices() {
      std::map<Polytope::Point, unsigned int> map;

      for (const Point &point : Vertices) {
         map[point]++;
      }

      return map.size();
   }

   void TriangleMesh::DeduplicateVertices() {

   }

   unsigned int TriangleMesh::RemoveDegenerateFaces() {
      // remove any face which has two or more identical vertices

      std::vector<unsigned int> faceIndicesToDelete;

      unsigned int index = 0;

      for (const Point3ui &face : Faces) {
         const Point v0 = Vertices[face.x];
         const Point v1 = Vertices[face.y];
         const Point v2 = Vertices[face.z];

         if (v0 == v1 || v1 == v2 || v2 == v0)
            faceIndicesToDelete.push_back(index);

         index++;
      }

      // delete old faces

      if (!faceIndicesToDelete.empty()) {

         std::sort(faceIndicesToDelete.begin(), faceIndicesToDelete.end());


         std::vector<Point3ui> newFaceVector;
         newFaceVector.reserve(Faces.size() - faceIndicesToDelete.size());

         index = 0;

         for (unsigned int i = 0; i < Faces.size(); i++) {
            if (i != faceIndicesToDelete[index]) {
               newFaceVector.push_back(Faces[i]);
            }
            else {
               index++;
            }
         }

         Faces = newFaceVector;
      }
   }

   unsigned int TriangleMesh::CountOrphanedVertices() {

      /// 1 if the vertex appears in a face, 0 if it doesn't (i.e. if it's orphaned)
      std::vector<bool> vertexState(Vertices.size(), true);
      for (const Point3ui &face : Faces) {
         vertexState[face.x] = false;
         vertexState[face.y] = false;
         vertexState[face.z] = false;
      }

      unsigned int sum = 0;
      for (auto && i : vertexState) {
         sum += i;
      }

      return sum;
   }

   bool TriangleMesh::Validate() {
      // each node must have either both children non-null xor at least 1 face
      if (root == nullptr)
         return true;

      std::queue<BVHNode *> q;
      q.push(root);

      bool valid = true;

      while (!q.empty()) {
         BVHNode* node = q.front();
         q.pop();

         bool lowNull = node->low == nullptr;
         bool highNull = node->high == nullptr;

         if (lowNull && !highNull) {
            valid = false;
            Log.WithTime("High child without low sibling :/");
         }

         if (!lowNull && highNull) {
            valid = false;
            Log.WithTime("Low child without high sibling :/");
         }

         if (lowNull && highNull && node->faces.empty()) {
            valid = false;
            Log.WithTime("Leaf node with empty faces :/");
         }

         if (!lowNull && !highNull && !node->faces.empty()) {
            valid = false;
            Log.WithTime("Interior node with faces :/");
         }

         if (!lowNull) {
            q.push(node->low);
         }
         if (!highNull) {
            q.push(node->high);
         }
      }

      return valid;
   }

   void BVHNode::ShrinkBoundingBox(const std::vector<Point> &vertices, const std::vector<Point3ui> &nodeFaces) {
      float minx = Polytope::FloatMax;
      float miny = Polytope::FloatMax;
      float minz = Polytope::FloatMax;

      float maxx = -Polytope::FloatMax;
      float maxy = -Polytope::FloatMax;
      float maxz = -Polytope::FloatMax;

      for (const Point3ui face : nodeFaces) {
         Point v0 = vertices[face.x];
         if (v0.x > maxx)
            maxx = v0.x;
         if (v0.x < minx)
            minx = v0.x;

         if (v0.y > maxy)
            maxy = v0.y;
         if (v0.y < miny)
            miny = v0.y;

         if (v0.z > maxz)
            maxz = v0.z;
         if (v0.z < minz)
            minz = v0.z;

         Point v1 = vertices[face.y];

         if (v1.x > maxx)
            maxx = v1.x;
         if (v1.x < minx)
            minx = v1.x;

         if (v1.y > maxy)
            maxy = v1.y;
         if (v1.y < miny)
            miny = v1.y;

         if (v1.z > maxz)
            maxz = v1.z;
         if (v1.z < minz)
            minz = v1.z;

         Point v2 = vertices[face.z];

         if (v2.x > maxx)
            maxx = v2.x;
         if (v2.x < minx)
            minx = v2.x;

         if (v2.y > maxy)
            maxy = v2.y;
         if (v2.y < miny)
            miny = v2.y;

         if (v2.z > maxz)
            maxz = v2.z;
         if (v2.z < minz)
            minz = v2.z;

      }

      bbox.p0.x = minx;
      bbox.p0.y = miny;
      bbox.p0.z = minz;

      bbox.p1.x = maxx;
      bbox.p1.y = maxy;
      bbox.p1.z = maxz;
   }
}
