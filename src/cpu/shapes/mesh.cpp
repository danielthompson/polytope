//
// Created by daniel on 5/2/20.
//

#include "mesh.h"
#include "mesh_intersect.h"

namespace poly {
   
   void Mesh::add_vertex(Point &v) {
      ObjectToWorld->ApplyInPlace(v);
      x_packed.push_back(v.x);
      y_packed.push_back(v.y);
      z_packed.push_back(v.z);
      num_vertices_packed++;
   }

   void Mesh::add_vertex(Point &v, Normal &n) {
      ObjectToWorld->ApplyInPlace(v);
      x_packed.push_back(v.x);
      y_packed.push_back(v.y);
      z_packed.push_back(v.z);

      ObjectToWorld->ApplyInPlace(n);
      nx_packed.push_back(n.x);
      ny_packed.push_back(n.y);
      nz_packed.push_back(n.z);
      
      num_vertices_packed++;
   }

   void Mesh::add_vertex(float vx, float vy, float vz) {
      ObjectToWorld->ApplyPoint(vx, vy, vz);
      x_packed.push_back(vx);
      y_packed.push_back(vy);
      z_packed.push_back(vz);
      num_vertices_packed++;
   }
   
   void Mesh::add_packed_face(const unsigned int v0_index, const unsigned int v1_index, const unsigned int v2_index) {
      fv0.push_back(v0_index);
      fv1.push_back(v1_index);
      fv2.push_back(v2_index);
      

      {
         const float p0x = x_packed[v0_index];
         BoundingBox->p0.x = p0x < BoundingBox->p0.x ? p0x : BoundingBox->p0.x;
         BoundingBox->p1.x = p0x > BoundingBox->p1.x ? p0x : BoundingBox->p1.x;
      }

      {
         const float p0y = y_packed[v0_index];
         BoundingBox->p0.y = p0y < BoundingBox->p0.y ? p0y : BoundingBox->p0.y;
         BoundingBox->p1.y = p0y > BoundingBox->p1.y ? p0y : BoundingBox->p1.y;
      }

      {
         const float p0z = z_packed[v0_index];
         BoundingBox->p0.z = p0z < BoundingBox->p0.z ? p0z : BoundingBox->p0.z;
         BoundingBox->p1.z = p0z > BoundingBox->p1.z ? p0z : BoundingBox->p1.z;
      }

      {
         const float p1x = x_packed[v1_index];
         BoundingBox->p0.x = p1x < BoundingBox->p0.x ? p1x : BoundingBox->p0.x;
         BoundingBox->p1.x = p1x > BoundingBox->p1.x ? p1x : BoundingBox->p1.x;
      }

      {
         const float p1y = y_packed[v1_index];
         BoundingBox->p0.y = p1y < BoundingBox->p0.y ? p1y : BoundingBox->p0.y;
         BoundingBox->p1.y = p1y > BoundingBox->p1.y ? p1y : BoundingBox->p1.y;
      }

      {
         const float p1z = z_packed[v1_index];
         BoundingBox->p0.z = p1z < BoundingBox->p0.z ? p1z : BoundingBox->p0.z;
         BoundingBox->p1.z = p1z > BoundingBox->p1.z ? p1z : BoundingBox->p1.z;
      }

      {
         const float p2x = x_packed[v2_index];
         BoundingBox->p0.x = p2x < BoundingBox->p0.x ? p2x : BoundingBox->p0.x;
         BoundingBox->p1.x = p2x > BoundingBox->p1.x ? p2x : BoundingBox->p1.x;
      }

      {
         const float p2y = y_packed[v2_index];
         BoundingBox->p0.y = p2y < BoundingBox->p0.y ? p2y : BoundingBox->p0.y;
         BoundingBox->p1.y = p2y > BoundingBox->p1.y ? p2y : BoundingBox->p1.y;
      }

      {
         const float p2z = z_packed[v2_index];
         BoundingBox->p0.z = p2z < BoundingBox->p0.z ? p2z : BoundingBox->p0.z;
         BoundingBox->p1.z = p2z > BoundingBox->p1.z ? p2z : BoundingBox->p1.z;
      }
      
      // calculate face normal
      const Point v0 = { x_packed[fv0[num_faces]], y_packed[fv0[num_faces]], z_packed[fv0[num_faces]] };
      const Point v1 = { x_packed[fv1[num_faces]], y_packed[fv1[num_faces]], z_packed[fv1[num_faces]] };
      const Point v2 = { x_packed[fv2[num_faces]], y_packed[fv2[num_faces]], z_packed[fv2[num_faces]] };

      const poly::Vector e0 = v1 - v0;
      const poly::Vector e1 = v2 - v1;
      poly::Vector plane_normal = e0.Cross(e1);
      plane_normal.Normalize();
      
      fnx.push_back(plane_normal.x);
      fny.push_back(plane_normal.y);
      fnz.push_back(plane_normal.z);

      num_faces++;
   }

   void Mesh::CalculateVertexNormals() {

      nx_packed = std::vector<float>(x_packed.size(), 0);
      ny_packed = std::vector<float>(y_packed.size(), 0);
      nz_packed = std::vector<float>(z_packed.size(), 0);
      
      for (unsigned int i = 0; i < num_faces; i++) {
         const float v0x = x_packed[fv0[i]];
         const float v0y = y_packed[fv0[i]];
         const float v0z = z_packed[fv0[i]];

         const float v1x = x_packed[fv1[i]];
         const float v1y = y_packed[fv1[i]];
         const float v1z = z_packed[fv1[i]];

         const float v2x = x_packed[fv2[i]];
         const float v2y = y_packed[fv2[i]];
         const float v2z = z_packed[fv2[i]];

         // step 1 - intersect with plane

         // const poly::Vector edge0 = vertex1 - vertex0;
         const float e0x = v1x - v0x;
         const float e0y = v1y - v0y;
         const float e0z = v1z - v0z;

         // const poly::Vector edge1 = vertex2 - vertex1;
         const float e1x = v2x - v1x;
         const float e1y = v2y - v1y;
         const float e1z = v2z - v1z;

         // poly::Vector planeNormal = edge0.Cross(edge1);
         float pnx = e0y * e1z - e0z * e1y;
         float pny = e0z * e1x - e0x * e1z;
         float pnz = e0x * e1y - e0y * e1x;

         // planeNormal.Normalize();         
         const float one_over_normal_length = 1.0f / std::sqrt(pnx * pnx + pny * pny + pnz * pnz);

         pnx *= one_over_normal_length;
         nx_packed[fv0[i]] += pnx;
         nx_packed[fv1[i]] += pnx;
         nx_packed[fv2[i]] += pnx;

         pny *= one_over_normal_length;
         ny_packed[fv0[i]] += pny;
         ny_packed[fv1[i]] += pny;
         ny_packed[fv2[i]] += pny;

         pnz *= one_over_normal_length;
         nz_packed[fv0[i]] += pnz;
         nz_packed[fv1[i]] += pnz;
         nz_packed[fv2[i]] += pnz;
      }

      nx = std::vector<float>(x.size(), 0);
      ny = std::vector<float>(y.size(), 0);
      nz = std::vector<float>(z.size(), 0);
      
      for (unsigned int i = 0; i < num_vertices_packed; i++) {
         const float nx_packed_sq = nx_packed[i] * nx_packed[i];
         const float ny_packed_sq = ny_packed[i] * ny_packed[i];
         const float nz_packed_sq = nz_packed[i] * nz_packed[i];
         
         const float one_over_normal_length = 1.0f / std::sqrt(nx_packed_sq + ny_packed_sq + nz_packed_sq);
         
         nx_packed[i] *= one_over_normal_length;
         ny_packed[i] *= one_over_normal_length;
         nz_packed[i] *= one_over_normal_length;
      }

      for (unsigned int i = 0; i < num_faces; i++) {
         unsigned int index = fv0[i];
         nx.push_back(nx_packed[index]);
         ny.push_back(ny_packed[index]);
         nz.push_back(nz_packed[index]);
      }

      for (unsigned int i = 0; i < num_faces; i++) {
         unsigned int index = fv1[i];
         nx.push_back(nx_packed[index]);
         ny.push_back(ny_packed[index]);
         nz.push_back(nz_packed[index]);
      }

      for (unsigned int i = 0; i < num_faces; i++) {
         unsigned int index = fv2[i];
         nx.push_back(nx_packed[index]);
         ny.push_back(ny_packed[index]);
         nz.push_back(nz_packed[index]);
      }
   }

   void Mesh::intersect(
         poly::Ray& world_ray, 
         poly::Intersection& intersection, 
         const unsigned int* face_indices, 
         const unsigned int num_face_indices) {
      //float t = poly::FloatMax;
      unsigned int hit_face_index = 0;
      bool hits = false;

      for (unsigned int face_index_index = 0; face_index_index < num_face_indices; face_index_index++) {
         
         unsigned int face_index = face_indices[face_index_index];

         const poly::Vector plane_normal = { fnx[face_index], fny[face_index], fnz[face_index] };
         const float divisor = plane_normal.Dot(world_ray.Direction);

         if (divisor == 0.0f) {
            // parallel
            continue;
         }
         
         const Point v0 = { x_packed[fv0[face_index]], y_packed[fv0[face_index]], z_packed[fv0[face_index]] };
         
//         poly::Vector plane_normal = e0.Cross(e1);
//         plane_normal.Normalize();

         const float ft = plane_normal.Dot(v0 - world_ray.Origin) / divisor;

         if (ft <= 0 || ft > world_ray.MinT) {
            continue;
         }

         const Point v1 = { x_packed[fv1[face_index]], y_packed[fv1[face_index]], z_packed[fv1[face_index]] };
         const Point v2 = { x_packed[fv2[face_index]], y_packed[fv2[face_index]], z_packed[fv2[face_index]] };

         const poly::Vector e0 = v1 - v0;
         const poly::Vector e1 = v2 - v1;
         const poly::Vector e2 = v0 - v2;
         
         // TODO fix this imprecise garbage
         const poly::Point hit_point = world_ray.GetPointAtT(ft);
         
         const poly::Vector p0 = hit_point - v0;
         const poly::Vector cross0 = e0.Cross(p0);
         const float normal0 = cross0.Dot(plane_normal);
         const bool pos0 = normal0 > 0;

         if (!pos0)
            continue;

         const poly::Vector p1 = hit_point - v1;
         const poly::Vector cross1 = e1.Cross(p1);
         const float normal1 = cross1.Dot(plane_normal);
         const bool pos1 = normal1 > 0;

         if (!pos1)
            continue;

         const poly::Vector p2 = hit_point - v2;
         const poly::Vector cross2 = e2.Cross(p2);
         const float normal2 = cross2.Dot(plane_normal);
         const bool pos2 = normal2 > 0;

         if (!pos2)
            continue;

         // hits
         world_ray.MinT = ft;
         hit_face_index = face_index;
         hits = true;
      }

      if (!hits/* || world_ray.MinT <= t*/) {
         return;
      }

//      bool debug = false;
//      if (worldSpaceRay.x == 245 && worldSpaceRay.y == 64) {
//         debug = true;
//         printf("t: %f\n", t);
//      }

      // now we have the closest face, if any

      intersection.Hits = true;
      //world_ray.MinT = t;
      intersection.faceIndex = hit_face_index;
      intersection.Location = world_ray.GetPointAtT(world_ray.MinT);

      const unsigned int v0_index = fv0[hit_face_index];
      const unsigned int v1_index = fv1[hit_face_index];
      const unsigned int v2_index = fv2[hit_face_index];
      
      const Point v0 = Point(x_packed[v0_index], y_packed[v0_index], z_packed[v0_index]);
      const Point v1 = Point(x_packed[v1_index], y_packed[v1_index], z_packed[v1_index]);
      const Point v2 = Point(x_packed[v2_index], y_packed[v2_index], z_packed[v2_index]);
//      
//      const Normal n0 = Normal(nx_packed[v0_index], ny_packed[v0_index], nz_packed[v0_index]);
//      const Normal n1 = Normal(nx_packed[v1_index], ny_packed[v1_index], nz_packed[v1_index]);
//      const Normal n2 = Normal(nx_packed[v2_index], ny_packed[v2_index], nz_packed[v2_index]);
//
//      // idea - use squared distance from intersection to each vertex to calculate vertex normal's weight for the interpolation
//      float squared_dist_0 = intersection.Location.Dot(v0);
//      float squared_dist_1 = intersection.Location.Dot(v1);
//      float squared_dist_2 = intersection.Location.Dot(v2);
//
//      float total_weight = squared_dist_0 + squared_dist_1 + squared_dist_2;
//      poly::Vector weights = {
//            total_weight / squared_dist_0, 
//            total_weight / squared_dist_1, 
//            total_weight / squared_dist_2 
//      };
//      weights.Normalize();
//
//      Normal n = {n0.x * weights.x + n1.x + weights.y + n2.x + weights.z,
//                  n0.y * weights.x + n1.y + weights.y + n2.y + weights.z,
//                  n0.z * weights.x + n1.z + weights.y + n2.z + weights.z};
//      n.Normalize();
      
      const poly::Vector edge0 = v1 - v0;
      const poly::Vector edge1 = v2 - v1;
      const poly::Vector edge2 = v0 - v2;
      //const poly::Vector planeNormal = {edge0.Cross(edge1);}
      poly::Normal plane_normal = { fnx[hit_face_index], fny[hit_face_index], fnz[hit_face_index] };

      // flip normal if needed
      if (world_ray.Direction.Dot(plane_normal) > 0) {
         plane_normal.Flip();
      }

      //n.Normalize();

      intersection.Normal = plane_normal;
      intersection.Shape = this;

      const float edge0dot = std::abs(edge0.Dot(edge1));
      const float edge1dot = std::abs(edge1.Dot(edge2));
      const float edge2dot = std::abs(edge2.Dot(edge0));

      if (edge0dot > edge1dot && edge0dot > edge2dot) {
         intersection.Tangent1 = edge0;
      } else if (edge1dot > edge0dot && edge1dot > edge2dot) {
         intersection.Tangent1 = edge1;
      } else {
         intersection.Tangent1 = edge2;
      }

      intersection.Tangent2 = intersection.Tangent1.Cross(intersection.Normal);

      intersection.Tangent1.Normalize();
      intersection.Tangent2.Normalize();
   }
   
   void Mesh::intersect(poly::Ray& world_ray, poly::Intersection& intersection, const std::vector<unsigned int>& face_indices) {
      intersect(world_ray, intersection, &face_indices[0], face_indices.size());
   }
   
   void Mesh::intersect(poly::Ray& worldSpaceRay, poly::Intersection& intersection) {
      float t = poly::FloatMax;
      unsigned int face_index = 0;
      bool hits = false;
      
      ispc::soa_linear_intersect(
            &x[0],
            &y[0],
            &z[0],
            worldSpaceRay.Origin.x,
            worldSpaceRay.Origin.y,
            worldSpaceRay.Origin.z,
            worldSpaceRay.Direction.x,
            worldSpaceRay.Direction.y,
            worldSpaceRay.Direction.z,
            t,
            face_index,
            hits,
            num_faces/*,
            worldSpaceRay.x,
            worldSpaceRay.y,
            worldSpaceRay.bounce*/);

      if (!hits || worldSpaceRay.MinT <= t) {
         return;
      }

//      bool debug = false;
//      if (worldSpaceRay.x == 245 && worldSpaceRay.y == 64) {
//         debug = true;
//         printf("t: %f\n", t);
//      }
      
      
      
      // now we have the closest face, if any

      intersection.Hits = true;
      worldSpaceRay.MinT = t;
      intersection.faceIndex = face_index;
      intersection.Location = worldSpaceRay.GetPointAtT(t);

      const unsigned int v0_index = fv0[face_index];
      const unsigned int v1_index = fv1[face_index];
      const unsigned int v2_index = fv2[face_index];

      const Point v0 = Point(x_packed[v0_index], y_packed[v0_index], z_packed[v0_index]);
      const Point v1 = Point(x_packed[v1_index], y_packed[v1_index], z_packed[v1_index]);
      const Point v2 = Point(x_packed[v2_index], y_packed[v2_index], z_packed[v2_index]);

      const poly::Vector edge0 = v1 - v0;
      const poly::Vector edge1 = v2 - v1;
      const poly::Vector edge2 = v0 - v2;
      const poly::Vector planeNormal = edge0.Cross(edge1);

      // flip normal if needed
      poly::Normal n(planeNormal.x, planeNormal.y, planeNormal.z);
      if (worldSpaceRay.Direction.Dot(n) > 0) {
         n.Flip();
      }

      n.Normalize();

      intersection.Normal = n;
      intersection.Shape = this;

      const float edge0dot = std::abs(edge0.Dot(edge1));
      const float edge1dot = std::abs(edge1.Dot(edge2));
      const float edge2dot = std::abs(edge2.Dot(edge0));

      if (edge0dot > edge1dot && edge0dot > edge2dot) {
         intersection.Tangent1 = edge0;
      } else if (edge1dot > edge0dot && edge1dot > edge2dot) {
         intersection.Tangent1 = edge1;
      } else {
         intersection.Tangent1 = edge2;
      }

      intersection.Tangent2 = intersection.Tangent1.Cross(intersection.Normal);

      intersection.Tangent1.Normalize();
      intersection.Tangent2.Normalize();
   }

   Point Mesh::random_surface_point() const {
      // TODO 1. generate a random point on a face instead of just using a vertex
      // TODO 2. weight face choice by face surface area

      const unsigned int index = RandomUniformBetween(0u, num_faces - 1);
      return Point(x[index], y[index], z[index]);
   }

   void Mesh::unpack_faces() {
      x.reserve(num_faces * 3);
      y.reserve(num_faces * 3);
      z.reserve(num_faces * 3);

      nx.reserve(num_faces * 3);
      ny.reserve(num_faces * 3);
      nz.reserve(num_faces * 3);

      // for each face
      for (unsigned int i = 0; i < num_faces; i++) {
         const unsigned int index = fv0[i];

         x.push_back(x_packed[index]);
         y.push_back(y_packed[index]);
         z.push_back(z_packed[index]);
      }

      for (unsigned int i = 0; i < num_faces; i++) {
         unsigned int index = fv1[i];

         x.push_back(x_packed[index]);
         y.push_back(y_packed[index]);
         z.push_back(z_packed[index]);

//         nx_expanded.push_back(nx[index]);
//         ny_expanded.push_back(ny[index]);
//         nz_expanded.push_back(nz[index]);
      }

      for (unsigned int i = 0; i < num_faces; i++) {
         unsigned int index = fv2[i];

         x.push_back(x_packed[index]);
         y.push_back(y_packed[index]);
         z.push_back(z_packed[index]);

//         nx_expanded.push_back(nx[index]);
//         ny_expanded.push_back(ny[index]);
//         nz_expanded.push_back(nz[index]);
      }

      num_vertices = 3 * num_faces;
      
   }

   Point Mesh::get_vertex(const unsigned int i) const {
      return { x_packed[i], y_packed[i], z_packed[i] };
   }

   Point3ui Mesh::get_vertex_indices_for_face(const unsigned int i) const {
      return { fv0[i], fv1[i], fv2[i] };
   }

   Mesh::~Mesh() {

   }

   bool Mesh::hits(const Ray &world_ray, const unsigned int *face_indices, unsigned int num_face_indices) const {
      for (unsigned int face_index = 0; face_index < num_face_indices; face_index++) {

         const Point v0 = { x_packed[fv0[face_index]], y_packed[fv0[face_index]], z_packed[fv0[face_index]] };
         const Point v1 = { x_packed[fv1[face_index]], y_packed[fv1[face_index]], z_packed[fv1[face_index]] };
         const Point v2 = { x_packed[fv2[face_index]], y_packed[fv2[face_index]], z_packed[fv2[face_index]] };

         const poly::Vector e0 = v1 - v0;
         const poly::Vector e1 = v2 - v1;
         poly::Vector plane_normal = e0.Cross(e1);
         plane_normal.Normalize();

         const float divisor = plane_normal.Dot(world_ray.Direction);

         if (divisor == 0.0f) {
            // parallel
            continue;
         }

         const float t = plane_normal.Dot(v0 - world_ray.Origin) / divisor;

         if (t <= 0) {
            continue;
         }
         // TODO fix this imprecise garbage
         const poly::Point hit_point = world_ray.GetPointAtT(t);

         const poly::Vector e2 = v0 - v2;
         const poly::Vector p0 = hit_point - v0;
         const poly::Vector cross0 = e0.Cross(p0);
         const float normal0 = cross0.Dot(plane_normal);
         const bool pos0 = normal0 > 0;

         if (!pos0)
            continue;

         const poly::Vector p1 = hit_point - v1;
         const poly::Vector cross1 = e1.Cross(p1);
         const float normal1 = cross1.Dot(plane_normal);
         const bool pos1 = normal1 > 0;

         if (!pos1)
            continue;

         const poly::Vector p2 = hit_point - v2;
         const poly::Vector cross2 = e2.Cross(p2);
         const float normal2 = cross2.Dot(plane_normal);
         const bool pos2 = normal2 > 0;

         if (!pos2)
            continue;

         // hits
         return true;
      }

      return false;
   }
   
   bool Mesh::hits(const Ray& world_ray, const std::vector<unsigned int>& face_indices) {
      return hits(world_ray, &face_indices[0], face_indices.size());
   }

}

