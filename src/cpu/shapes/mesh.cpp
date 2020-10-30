//
// Created by daniel on 5/2/20.
//

#include <atomic>
#include <cassert>
#include "mesh.h"
#include "mesh_intersect.h"
#include "../structures/stats.h"

#define assert_eq(a, b) do { if (a != b) fprintf(stderr, "a %f != b %f", a, b); assert(a == b); } while (0)

#define assert_lte(a, b) do { \
   if (a > b) \
      fprintf(stderr, "a %f ! <= b %f\n", a, b); \
      assert(a <= b); } while (0)
      
#define assert_gte(a, b) do { \
   if (a < b) \
      fprintf(stderr, "a %f ! >= b %f\n", a, b); \
      assert(a >= b); } while (0)

//extern std::atomic<int> num_triangle_intersections;   
extern thread_local poly::stats thread_stats;
      
namespace poly {

   float determinant(const float a1, const float a2, const float b1, const float b2) {
      return a1 * b2 - b1 * a2;
   }

   inline float clamp(const float v, const float low, const float high) {
      if (v < low)
         return low;
      if (v > high)
         return high;
      return v;
   }

   inline float diff_product(float a, float b, float c, float d) {
      float cd = c * d;
      float err = std::fma(-c, d, cd);
      float dop = std::fma(a, b, -cd);
      return dop + err;
   }

   std::pair<float, float> solve_linear_2x2(
         const float a1,
         const float a2,
         const float b1,
         const float b2,
         const float c1,
         const float c2) {
      const float one_over_determinant = 1.f / diff_product(a1, b2, b1, a2);
      const float x_numerator = diff_product(c1, b2, b1, c2);
      const float y_numerator = diff_product(a1, c2, c1, a2);
      const float x = x_numerator * one_over_determinant;
      const float y = y_numerator * one_over_determinant;
      return {x, y};
   }

   void Mesh::add_vertex(Point &v) {
      if (ObjectToWorld != nullptr)
         ObjectToWorld->ApplyInPlace(v);
      x_packed.push_back(v.x);
      y_packed.push_back(v.y);
      z_packed.push_back(v.z);
      num_vertices_packed++;
   }

   void Mesh::add_vertex(Point &v, Normal &n) {

      if (ObjectToWorld != nullptr)
         ObjectToWorld->ApplyInPlace(v);
      x_packed.push_back(v.x);
      y_packed.push_back(v.y);
      z_packed.push_back(v.z);

      if (ObjectToWorld != nullptr)
         ObjectToWorld->ApplyInPlace(n);
      nx_packed.push_back(n.x);
      ny_packed.push_back(n.y);
      nz_packed.push_back(n.z);

      num_vertices_packed++;
   }

   void Mesh::add_vertex(float vx, float vy, float vz) {
      if (ObjectToWorld != nullptr)
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
      const Point v0 = {x_packed[fv0[num_faces]], y_packed[fv0[num_faces]], z_packed[fv0[num_faces]]};
      const Point v1 = {x_packed[fv1[num_faces]], y_packed[fv1[num_faces]], z_packed[fv1[num_faces]]};
      const Point v2 = {x_packed[fv2[num_faces]], y_packed[fv2[num_faces]], z_packed[fv2[num_faces]]};

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

   float dim(const Vector v, const int d) {
      switch (d) {
         case 0:
            return v.x;
         case 1:
            return v.y;
         default:
            return v.z;
      }
   }
   
   void Mesh::intersect(
         poly::Ray &world_ray,
         poly::Intersection &intersection,
         const unsigned int *face_indices,
         const unsigned int num_face_indices) const {

      thread_stats.num_triangle_intersections++;

      for (unsigned int face_index_index = 0; face_index_index < num_face_indices; face_index_index++) {

         unsigned int face_index = face_indices[face_index_index];

         const Point v0 = {x_packed[fv0[face_index]], y_packed[fv0[face_index]], z_packed[fv0[face_index]]};
         const Point v1 = {x_packed[fv1[face_index]], y_packed[fv1[face_index]], z_packed[fv1[face_index]]};
         const Point v2 = {x_packed[fv2[face_index]], y_packed[fv2[face_index]], z_packed[fv2[face_index]]};

         // wald watertight intersection
         // http://jcgt.org/published/0002/01/05/paper.pdf

         float dx_abs = fabsf(world_ray.Direction.x);
         float dy_abs = fabsf(world_ray.Direction.y);
         float dz_abs = fabsf(world_ray.Direction.z);

         int kz = 0;
         if (dy_abs > dx_abs && dy_abs > dz_abs)
            kz = 1;
         else if (dz_abs > dx_abs && dz_abs > dy_abs)
            kz = 2;

         int kx = kz + 1;
         if (kx == 3)
            kx = 0;
         int ky = kx + 1;
         if (ky == 3)
            ky = 0;
         
         // calculate vertices relative to ray origin
         const Vector A = v0 - world_ray.Origin;
         const Vector B = v1 - world_ray.Origin;
         const Vector C = v2 - world_ray.Origin;

         // swap kx and ky dimension to preserve winding direction of triangles
         if (dim(world_ray.Direction, kz) < 0.0f) {
            // swap(kx, ky)
            int temp = kx;
            kx = ky;
            ky = temp;
         }

         // calculate shear constants
         float Sx = dim(world_ray.Direction, kx) / dim(world_ray.Direction, kz);
         float Sy = dim(world_ray.Direction, ky) / dim(world_ray.Direction, kz);
         float Sz = 1.0f / dim(world_ray.Direction, kz);

         // perform shear and scale of vertices
         const float Ax = dim(A, kx) - Sx * dim(A, kz);
         const float Ay = dim(A, ky) - Sy * dim(A, kz);
         const float Bx = dim(B, kx) - Sx * dim(B, kz);
         const float By = dim(B, ky) - Sy * dim(B, kz);
         const float Cx = dim(C, kx) - Sx * dim(C, kz);
         const float Cy = dim(C, ky) - Sy * dim(C, kz);

         // calculate scaled barycentric coordinates
         float U = Cx * By - Cy * Bx;
         float V = Ax * Cy - Ay * Cx;
         float W = Bx * Ay - By * Ax;

         // fallback to test against edges using double precision
         if (U == 0.0f || V == 0.0f || W == 0.0f) {
            double CxBy = (double)Cx*(double)By;
            double CyBx = (double)Cy*(double)Bx;
            U = (float)(CxBy - CyBx);

            double AxCy = (double)Ax*(double)Cy;
            double AyCx = (double)Ay*(double)Cx;
            V = (float)(AxCy - AyCx);

            double BxAy = (double)Bx*(double)Ay;
            double ByAx = (double)By*(double)Ax;
            W = (float)(BxAy - ByAx);
         }

         // Perform edge tests. Moving this test before and at the end of the previous
         // conditional gives higher performance.
         // backface culling:
         if (U < 0.0f || V < 0.0f || W < 0.0f)
            continue;
         // no backface culling:
         //if ((U<0.0f || V<0.0f || W<0.0f) &&(U>0.0f || V>0.0f || W>0.0f)) return;

         // calculate determinant
         float det = U + V + W;
         if (det == 0.0f)
            continue;

         // calculate scaled z-coordinates of vertices and use them to calculate the hit distance
         const float Az = Sz * dim(A, kz);
         const float Bz = Sz * dim(B, kz);
         const float Cz = Sz * dim(C, kz);
         const float T = U * Az + V * Bz + W * Cz;

         // backface culling
         if (T < 0.0f || T > world_ray.MinT * det)
            continue;

         // no backface culling
         // int det_sign = sign_mask(det);if (xorf(T,det_sign) < 0.0f) ||xorf(T,det_sign) > hit.t*xorf(det, det_sign))return;

         // normalize
         const float rcpDet = 1.0f / det;
         float u = U * rcpDet;
         float v = V * rcpDet;
         float w = W * rcpDet;
         world_ray.MinT = T * rcpDet;
         intersection.Hits = true;
         intersection.faceIndex = face_index;
         intersection.Shape = this;
         intersection.Location = Point(v0.x * u + v1.x * v + v2.x * w,
                                       v0.y * u + v1.y * v + v2.y * w,
                                       v0.z * u + v1.z * v + v2.z * w);
         intersection.u = u;
         intersection.v = v;
         intersection.w = w;
      }

      if (!intersection.Hits) {
         return;
      }

      thread_stats.num_triangle_intersections_hit++;

      intersection.Hits = true;

      // TODO refactor this to do it only once after all faces/bvh nodes are intersected
      const unsigned int v1_index = intersection.faceIndex + num_faces;
      const unsigned int v2_index = intersection.faceIndex + num_faces * 2;
      
      const Point v0 = {x[intersection.faceIndex], y[intersection.faceIndex], z[intersection.faceIndex]};
      const Point v1 = {x[v1_index], y[v1_index], z[v1_index]};
      const Point v2 = {x[v2_index], y[v2_index], z[v2_index]};

      // edge functions
      const Vector e0 = v1 - v0;
      const Vector e1 = v2 - v1;
      const Vector e2 = v0 - v2;
      
      Normal n;
      
      if (this->has_vertex_normals) {
         const Vector v0n = {nx[intersection.faceIndex], ny[intersection.faceIndex],nz[intersection.faceIndex]};
         const Vector v1n = {nx[v1_index], ny[v1_index],nz[v1_index]};
         const Vector v2n = {nx[v2_index], ny[v2_index],nz[v2_index]};
         n = {v0n.x * intersection.u + v1n.x * intersection.v + v2n.x * intersection.w,
              v0n.y * intersection.u + v1n.y * intersection.v + v2n.y * intersection.w,
              v0n.z * intersection.u + v1n.z * intersection.v + v2n.z * intersection.w
         };
      }
      else {
         const Vector v = e0.Cross(e1);
         n = {v.x, v.y, v.z};
      }

      const float ray_dot_normal = world_ray.Direction.Dot(n);
      const float flip_factor = ray_dot_normal > 0 ? -1 : 1;
      n = n * flip_factor;
      n.Normalize();
      intersection.bent_normal = n;
      intersection.geo_normal = n;
      const float edge0dot = fabsf(e0.Dot(e1));
      const float edge1dot = fabsf(e1.Dot(e2));
      const float edge2dot = fabsf(e2.Dot(e0));

      if (edge0dot > edge1dot && edge0dot > edge2dot) {
         intersection.Tangent1 = e0;
      } else if (edge1dot > edge0dot && edge1dot > edge2dot) {
         intersection.Tangent1 = e1;
      } else {
         intersection.Tangent1 = e2;
      }

      intersection.Tangent2 = intersection.Tangent1.Cross(n);
      intersection.Tangent1.Normalize();
      intersection.Tangent2.Normalize();
      
   }

   void Mesh::intersect(poly::Ray &world_ray, poly::Intersection &intersection,
                        const std::vector<unsigned int> &face_indices) {
      intersect(world_ray, intersection, &face_indices[0], face_indices.size());
   }

   void Mesh::intersect(poly::Ray &worldSpaceRay, poly::Intersection &intersection) {
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

      intersection.geo_normal = n;
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

      intersection.Tangent2 = intersection.Tangent1.Cross(intersection.geo_normal);

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

      if (has_vertex_normals) {
         nx.reserve(num_faces * 3);
         ny.reserve(num_faces * 3);
         nz.reserve(num_faces * 3);
      }
      
      // for each face
      for (unsigned int i = 0; i < num_faces; i++) {
         const unsigned int index = fv0[i];

         x.push_back(x_packed[index]);
         y.push_back(y_packed[index]);
         z.push_back(z_packed[index]);

         if (has_vertex_normals) {
            nx.push_back(nx_packed[index]);
            ny.push_back(ny_packed[index]);
            nz.push_back(nz_packed[index]);
         }
      }

      for (unsigned int i = 0; i < num_faces; i++) {
         unsigned int index = fv1[i];

         x.push_back(x_packed[index]);
         y.push_back(y_packed[index]);
         z.push_back(z_packed[index]);

         if (has_vertex_normals) {
            nx.push_back(nx_packed[index]);
            ny.push_back(ny_packed[index]);
            nz.push_back(nz_packed[index]);
         }
      }

      for (unsigned int i = 0; i < num_faces; i++) {
         unsigned int index = fv2[i];

         x.push_back(x_packed[index]);
         y.push_back(y_packed[index]);
         z.push_back(z_packed[index]);

         if (has_vertex_normals) {
            nx.push_back(nx_packed[index]);
            ny.push_back(ny_packed[index]);
            nz.push_back(nz_packed[index]);
         }
      }

      num_vertices = 3 * num_faces;

   }

   Point Mesh::get_vertex(const unsigned int i) const {
      return {x_packed[i], y_packed[i], z_packed[i]};
   }

   void Mesh::get_vertices_for_face(unsigned int i, poly::Point vertices[3]) const {
      vertices[0] = { x_packed[fv0[i]], y_packed[fv0[i]], z_packed[fv0[i]] };
      vertices[1] = { x_packed[fv1[i]], y_packed[fv1[i]], z_packed[fv1[i]] };
      vertices[2] = { x_packed[fv2[i]], y_packed[fv2[i]], z_packed[fv2[i]] };
   }
   
   Point3ui Mesh::get_vertex_indices_for_face(const unsigned int i) const {
      return {fv0[i], fv1[i], fv2[i]};
   }

   Mesh::~Mesh() {

   }

   bool Mesh::hits(const Ray &world_ray, const unsigned int *face_indices, unsigned int num_face_indices) const {
      for (unsigned int face_index = 0; face_index < num_face_indices; face_index++) {

         const Point v0 = {x_packed[fv0[face_index]], y_packed[fv0[face_index]], z_packed[fv0[face_index]]};
         const Point v1 = {x_packed[fv1[face_index]], y_packed[fv1[face_index]], z_packed[fv1[face_index]]};
         const Point v2 = {x_packed[fv2[face_index]], y_packed[fv2[face_index]], z_packed[fv2[face_index]]};

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

   bool Mesh::hits(const Ray &world_ray, const std::vector<unsigned int> &face_indices) {
      return hits(world_ray, &face_indices[0], face_indices.size());
   }

   float Mesh::surface_area(const unsigned int face_index) const {

      const poly::Point p0 = {x_packed[fv0[face_index]], y_packed[fv0[face_index]], z_packed[fv0[face_index]]};
      const poly::Point p1 = {x_packed[fv1[face_index]], y_packed[fv1[face_index]], z_packed[fv1[face_index]]};
      const poly::Point p2 = {x_packed[fv2[face_index]], y_packed[fv2[face_index]], z_packed[fv2[face_index]]};

      const Vector e0 = p0 - p1;
      const Vector e1 = p1 - p2;
      const Vector e2 = p2 - p0;

      float a = e0.Length();
      float b = e1.Length();
      float c = e2.Length();

      if (a < b)
         std::swap(a, b);
      if (a < c)
         std::swap(a, c);
      if (b < c)
         std::swap(b, c);

      assert(a >= b);
      assert(b >= c);
      
      if (c - (a - b) < 0.f) {
         fprintf(stderr, "triangle side length fail :/\n");
         exit(1);
      }

      const float sa = std::sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)));

      return sa;
   }
}
