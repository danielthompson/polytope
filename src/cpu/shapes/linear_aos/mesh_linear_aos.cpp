//
// Created by daniel on 5/3/20.
//

#include "mesh_linear_aos.h"

namespace poly {

   void MeshLinearAOS::intersect(Ray &ray, Intersection *intersection) {
      unsigned int faceIndex = 0;
      // TODO delete this
      std::vector<float> t_values;

      unsigned int numInside = 0;

      for (const Point3ui &face : Faces) {
         const Point vertex0 = Vertices[face.x];
         const Point vertex1 = Vertices[face.y];
         const Point vertex2 = Vertices[face.z];

         // step 1 - intersect with plane

         const poly::Vector edge0 = vertex1 - vertex0;
         const poly::Vector edge1 = vertex2 - vertex1;

         poly::Vector planeNormal = edge0.Cross(edge1);
         planeNormal.Normalize();

         const float divisor = planeNormal.Dot(ray.Direction);
         if (divisor == 0.0f) {
            // parallel
            t_values.push_back(-1);
            faceIndex++;
            continue;
         }

         const float t = planeNormal.Dot(vertex0 - ray.Origin) / divisor;
         if (t <= 0) {
            t_values.push_back(-1);
            faceIndex++;
            continue;
         }

         t_values.push_back(t);

         const poly::Point hitPoint = ray.GetPointAtT(t);

         // step 2 - inside/outside test

         const poly::Vector edge2 = vertex0 - vertex2;

         const poly::Vector p0 = hitPoint - vertex0;
         const poly::Vector cross0 = edge0.Cross(p0);
         const float normal0 = cross0.Dot(planeNormal);
         const bool pos0 = normal0 > 0;

         // 4. Repeat for all 3 edges
         const poly::Vector p1 = hitPoint - vertex1;
         const poly::Vector cross1 = edge1.Cross(p1);
         const float normal1 = cross1.Dot(planeNormal);
         const bool pos1 = normal1 > 0;

         const poly::Vector p2 = hitPoint - vertex2;
         const poly::Vector cross2 = edge2.Cross(p2);
         const float normal2 = cross2.Dot(planeNormal);
         const bool pos2 = normal2 > 0;

         bool inside = pos0 && pos1 && pos2;

         if (inside)
            numInside++;

         if (inside && t < ray.MinT) {
            intersection->Hits = true;
            ray.MinT = t;
            intersection->faceIndex = faceIndex;
            intersection->Location = hitPoint;

            // flip normal if needed
            poly::Normal n(planeNormal.x, planeNormal.y, planeNormal.z);
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
            } else if (edge1dot > edge0dot && edge1dot > edge2dot) {
               intersection->Tangent1 = edge1;
               intersection->Tangent2 = edge2;
            } else {
               intersection->Tangent1 = edge2;
               intersection->Tangent2 = edge0;
            }

            intersection->Tangent1.Normalize();
            intersection->Tangent2.Normalize();
         }
         faceIndex++;
      }
   }

   void MeshLinearAOS::add_vertex(float x, float y, float z) {
      ObjectToWorld->ApplyPoint(x, y, z);
      Vertices.emplace_back(x, y, z);
      num_vertices_packed++;
   }

   void MeshLinearAOS::add_vertex(Point &v) {
      ObjectToWorld->ApplyInPlace(v);
      Vertices.push_back(v);
      num_vertices_packed++;
   }

   void MeshLinearAOS::add_packed_face(const unsigned int v0, const unsigned int v1, const unsigned int v2) {
      Faces.emplace_back(v0, v1, v2);
      num_faces++;
   }

   Point MeshLinearAOS::get_vertex(const unsigned int i) const {
      return Vertices[i];
   }

   Point3ui MeshLinearAOS::get_vertex_indices_for_face(const unsigned int i) const {
      return Faces[i];
   }

   void MeshLinearAOS::CalculateVertexNormals() {
      // TODO
   }

   Point MeshLinearAOS::random_surface_point() const {
      // TODO
      return Point();
   }

   void MeshLinearAOS::unpack_faces() {
      num_vertices = num_vertices_packed;
   }
}