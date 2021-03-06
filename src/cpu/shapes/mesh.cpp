//
// Created by daniel on 5/2/20.
//

#include <atomic>
#include <cassert>
#include "mesh.h"
#include "mesh_intersect.h"
#include "../structures/stats.h"
#include "../../common/utilities/Common.h"

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

   void mesh_geometry::add_vertex(Point &v) {
      x_packed.push_back(v.x);
      y_packed.push_back(v.y);
      z_packed.push_back(v.z);
      num_vertices_packed++;
   }

   void mesh_geometry::add_vertex(Point &v, Normal &n) {
      x_packed.push_back(v.x);
      y_packed.push_back(v.y);
      z_packed.push_back(v.z);
      nx_packed.push_back(n.x);
      ny_packed.push_back(n.y);
      nz_packed.push_back(n.z);
      num_vertices_packed++;
   }

   void mesh_geometry::add_vertex(Point &p, const float u, const float v) {
      x_packed.push_back(p.x);
      y_packed.push_back(p.y);
      z_packed.push_back(p.z);
      u_packed.push_back(u);
      v_packed.push_back(v);
      num_vertices_packed++;
   }

   void mesh_geometry::add_vertex(Point &p, Normal &n, const float u, const float v) {
      x_packed.push_back(p.x);
      y_packed.push_back(p.y);
      z_packed.push_back(p.z);
      nx_packed.push_back(n.x);
      ny_packed.push_back(n.y);
      nz_packed.push_back(n.z);
      u_packed.push_back(u);
      v_packed.push_back(v);
      num_vertices_packed++;
   }

   void mesh_geometry::add_vertex(float vx, float vy, float vz) {
      x_packed.push_back(vx);
      y_packed.push_back(vy);
      z_packed.push_back(vz);
      num_vertices_packed++;
   }

   void mesh_geometry::add_packed_face(const unsigned int v0_index, const unsigned int v1_index, const unsigned int v2_index) {
      fv0.push_back(v0_index);
      fv1.push_back(v1_index);
      fv2.push_back(v2_index);

      // calculate face normal
      // TODO we can't do this here, because depending on the order of the input file,
      // the vertices might not have been defined yet
//      const Point v0 = {x_packed[fv0[num_faces]], y_packed[fv0[num_faces]], z_packed[fv0[num_faces]]};
//      const Point v1 = {x_packed[fv1[num_faces]], y_packed[fv1[num_faces]], z_packed[fv1[num_faces]]};
//      const Point v2 = {x_packed[fv2[num_faces]], y_packed[fv2[num_faces]], z_packed[fv2[num_faces]]};
//
//      const poly::Vector e0 = v1 - v0;
//      const poly::Vector e1 = v2 - v1;
//      poly::Vector plane_normal = e0.Cross(e1);
//      plane_normal.Normalize();
//
//      fnx.push_back(plane_normal.x);
//      fny.push_back(plane_normal.y);
//      fnz.push_back(plane_normal.z);

      num_faces++;
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
   

   
   poly::Vector abs(const poly::Vector& v) {
      return {std::abs(v.x), std::abs(v.y), std::abs(v.z)};
   }
   
   float max_component(const poly::Vector& v) {
      if (v.x >= v.y && v.x >= v.z)
         return v.x;
      else if (v.y >= v.x && v.y >= v.z)
         return v.y;
      return v.z; 
   }
   
   poly::Vector permute(const poly::Vector& v, const int order[3]) {
      return { v[order[0]], v[order[1]], v[order[2]]};
   }
   
   void Mesh::intersect(
         poly::Ray &world_ray,
         poly::Intersection &intersection,
         const unsigned int *face_indices,
         const unsigned int num_face_indices) const {

      thread_stats.num_triangle_intersections++;

      for (unsigned int face_index_index = 0; face_index_index < num_face_indices; face_index_index++) {

         unsigned int face_index = face_indices[face_index_index];

         Point v0 = {mesh_geometry->x_packed[mesh_geometry->fv0[face_index]],
                     mesh_geometry->y_packed[mesh_geometry->fv0[face_index]],
                     mesh_geometry->z_packed[mesh_geometry->fv0[face_index]]};
         Point v1 = {mesh_geometry->x_packed[mesh_geometry->fv1[face_index]],
                     mesh_geometry->y_packed[mesh_geometry->fv1[face_index]],
                     mesh_geometry->z_packed[mesh_geometry->fv1[face_index]]};
         Point v2 = {mesh_geometry->x_packed[mesh_geometry->fv2[face_index]],
                     mesh_geometry->y_packed[mesh_geometry->fv2[face_index]],
                     mesh_geometry->z_packed[mesh_geometry->fv2[face_index]]};
         
         // transform to world space
         
         object_to_world->ApplyInPlace(v0);
         object_to_world->ApplyInPlace(v1);
         object_to_world->ApplyInPlace(v2);

         // wald watertight intersection
         // http://jcgt.org/published/0002/01/05/paper.pdf

         if (intersection.x == 128 && intersection.y == 128 && std::isnan(world_ray.Direction.x)) {
            continue;
         }
         
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
         Vector p0t = v0 - world_ray.Origin;
         Vector p1t = v1 - world_ray.Origin;
         Vector p2t = v2 - world_ray.Origin;
         
         // swap kx and ky dimension to preserve winding direction of triangles
//         if (dim(world_ray.Direction, kz) < 0.0f) {
//            // swap(kx, ky)
//            int temp = kx;
//            kx = ky;
//            ky = temp;
//         }

         int permutation[3] = {kx, ky, kz};
         Vector d = permute(world_ray.Direction, permutation);

         p0t = permute(p0t, permutation);
         p1t = permute(p1t, permutation);
         p2t = permute(p2t, permutation);
         
         // calculate shear constants
         float Sx = -d.x / d.z;
         float Sy = -d.y / d.z;
         float Sz = 1.f / d.z;

         // perform shear and scale of vertices
         p0t.x = std::fma(Sx, p0t.z, p0t.x);
         p0t.y = std::fma(Sy, p0t.z, p0t.y);
         p1t.x = std::fma(Sx, p1t.z, p1t.x);
         p1t.y = std::fma(Sy, p1t.z, p1t.y);
         p2t.x = std::fma(Sx, p2t.z, p2t.x);
         p2t.y = std::fma(Sy, p2t.z, p2t.y);

         // calculate scaled barycentric coordinates
         float U = difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
         float V = difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
         float W = difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);

         // fallback to test against edges using double precision
         if (U == 0.0f || V == 0.0f || W == 0.0f) {
            double CxBy = (double)p2t.x * (double)p1t.y;
            double CyBx = (double)p2t.y * (double)p1t.x;
            U = (float)(CyBx - CxBy);

            double AxCy = (double)p0t.x * (double)p2t.y;
            double AyCx = (double)p0t.y * (double)p2t.x;
            V = (float)(AyCx - AxCy);

            double BxAy = (double)p1t.x * (double)p0t.y;
            double ByAx = (double)p1t.y * (double)p0t.x;
            W = (float)(ByAx - BxAy);
         }

         // Perform edge tests. Moving this test before and at the end of the previous
         // conditional gives higher performance.
         // backface culling:
//          if (U < 0.0f || V < 0.0f || W < 0.0f)
//            continue;
         // no backface culling:
         if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) 
            continue;

         // calculate determinant
         float det = U + V + W;
         if (det == 0.0f)
            continue;

         // calculate scaled z-coordinates of vertices and use them to calculate the hit distance
         p0t.z *= Sz;
         p1t.z *= Sz;
         p2t.z *= Sz;
         
         const float T = std::fma(U, p0t.z, sum_of_products(V, p1t.z, W, p2t.z));

         // backface culling
//         bool backface = false;
//         if (T < 0.0f || T > world_ray.MinT * det)
//            continue;

         // no backface culling
         if (det < 0.f && (T >= 0 || T < world_ray.MinT * det)) {
            continue;
         }
         if (det > 0.f && (T <= 0 || T > world_ray.MinT * det)) {
            continue;
         }

         // normalize
         const float rcpDet = 1.0f / det;
         float u = U * rcpDet;
         float v = V * rcpDet;
         float w = W * rcpDet;
         const float computed_t = T * rcpDet;

         if (std::isnan(computed_t) && intersection.x == 128 && intersection.y == 128) {
            // but... why is it nan to begin with?
            continue;
         }
         
         float maxZt = max_component(abs(poly::Vector(p0t.z, p1t.z, p2t.z)));
         float deltaZ = poly::Gamma3 * maxZt;
         float maxXt = max_component(abs(poly::Vector(p0t.x, p1t.x, p2t.x)));
         float maxYt = max_component(abs(poly::Vector(p0t.y, p1t.y, p2t.y)));
         float deltaX = poly::Gamma5 * (maxXt + maxZt);
         float deltaY = poly::Gamma5 * (maxYt + maxZt);
         float deltaE = 2 * (poly::Gamma2 * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);
         float maxE = max_component(abs(poly::Vector(U, V, W)));
         float deltaT = 3 * (poly::Gamma3 * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) * std::abs(rcpDet);
         if (computed_t <= deltaT)
            continue;
         
         // http://www.pbr-book.org/3ed-2018/Shapes/Managing_Rounding_Error.html#x4-ParametricEvaluation:Triangles
         float x_error_sum = std::abs(u * v0.x) + std::abs(v * v1.x) + std::abs(w * v2.x);
         float y_error_sum = std::abs(u * v0.y) + std::abs(v * v1.y) + std::abs(w * v2.y);
         float z_error_sum = std::abs(u * v0.z) + std::abs(v * v1.z) + std::abs(w * v2.z);
         intersection.error = Vector(x_error_sum, y_error_sum, z_error_sum) * poly::Gamma7;

         
         world_ray.MinT = computed_t; 
         intersection.Hits = true;
         intersection.faceIndex = face_index;
         intersection.Shape = this;
         intersection.Location = Point(std::fmaf(v0.x, u, sum_of_products(v1.x, v, v2.x, w)),
                                       std::fmaf(v0.y, u, sum_of_products(v1.y, v, v2.y, w)),
                                       std::fmaf(v0.z, u, sum_of_products(v1.z, v, v2.z, w)));
         intersection.u = u;
         intersection.v = v;
         intersection.w = w;
         
         if (mesh_geometry->has_vertex_uvs) {
            float p0_u = mesh_geometry->u_packed[mesh_geometry->fv0[face_index]];
            float p0_v = mesh_geometry->v_packed[mesh_geometry->fv0[face_index]];
            
            float p1_u = mesh_geometry->u_packed[mesh_geometry->fv1[face_index]];
            float p1_v = mesh_geometry->v_packed[mesh_geometry->fv1[face_index]];
            
            float p2_u = mesh_geometry->u_packed[mesh_geometry->fv2[face_index]];
            float p2_v = mesh_geometry->v_packed[mesh_geometry->fv2[face_index]];
            
            intersection.u_tex_lerp = p0_u * u + p1_u * v + p2_u * w;
            intersection.v_tex_lerp = p0_v * u + p1_v * v + p2_v * w;
         }
      }

      if (!intersection.Hits) {
         return;
      }

      if (intersection.x == 128 && intersection.y == 128)
         Log.debug("Hits with t %f", world_ray.MinT);
      
      thread_stats.num_triangle_intersections_hit++;

      intersection.Hits = true;

      // TODO refactor this to do it only once after all faces/bvh nodes are intersected
      const unsigned int v1_index = intersection.faceIndex + mesh_geometry->num_faces;
      const unsigned int v2_index = intersection.faceIndex + mesh_geometry->num_faces * 2;
      
      Point v0 = {mesh_geometry->x[intersection.faceIndex], mesh_geometry->y[intersection.faceIndex], mesh_geometry->z[intersection.faceIndex]};
      Point v1 = {mesh_geometry->x[v1_index], mesh_geometry->y[v1_index], mesh_geometry->z[v1_index]};
      Point v2 = {mesh_geometry->x[v2_index], mesh_geometry->y[v2_index], mesh_geometry->z[v2_index]};

      object_to_world->ApplyInPlace(v0);
      object_to_world->ApplyInPlace(v1);
      object_to_world->ApplyInPlace(v2);
      
      // edge functions
      const Vector e0 = v1 - v0;
      const Vector e1 = v2 - v1;
      const Vector e2 = v0 - v2;
      
      Normal n;
      
      if (mesh_geometry->has_vertex_normals) {
         Normal v0n = {mesh_geometry->nx[intersection.faceIndex], 
                       mesh_geometry->ny[intersection.faceIndex],
                       mesh_geometry->nz[intersection.faceIndex]};
         Normal v1n = {mesh_geometry->nx[v1_index], 
                       mesh_geometry->ny[v1_index],
                       mesh_geometry->nz[v1_index]};
         Normal v2n = {mesh_geometry->nx[v2_index], 
                       mesh_geometry->ny[v2_index],
                       mesh_geometry->nz[v2_index]};
         
         object_to_world->ApplyInPlace(v0n);
         object_to_world->ApplyInPlace(v1n);
         object_to_world->ApplyInPlace(v2n);
         
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
      
      // offset ray origin along normal
      // http://www.pbr-book.org/3ed-2018/Shapes/Managing_Rounding_Error.html#RobustSpawnedRayOrigins
      poly::Vector abs_normal = { std::abs(n.x), std::abs(n.y), std::abs(n.z) };
      float d = intersection.error.Dot(abs_normal);
      poly::Vector offset = poly::Vector(n.x, n.y, n.z) * d;
      if (world_ray.Direction.Dot(n) > 0)
         offset = -offset;
      poly::Point new_p = intersection.Location + offset;
      for (int i = 0; i < 3; i++) {
         if (offset[i] > 0) {
            new_p[i] = std::nextafter(new_p[i], poly::Infinity);
         } 
         else if (offset[i] < 0) {
            new_p[i] = std::nextafter(new_p[i], -poly::Infinity);
         }
      }
      
      intersection.Location = new_p;
      
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
      bool debug = false;
      if (intersection.x == 128 && intersection.y == 128) {
         debug = true;   
      }
      
      intersection.Tangent1.Normalize();
      intersection.Tangent2.Normalize();
      if (intersection.x == 128 && intersection.y == 128 && std::isnan(intersection.Tangent2.x)) {
         debug = true;
      }
      
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
            &mesh_geometry->x[0],
            &mesh_geometry->y[0],
            &mesh_geometry->z[0],
            worldSpaceRay.Origin.x,
            worldSpaceRay.Origin.y,
            worldSpaceRay.Origin.z,
            worldSpaceRay.Direction.x,
            worldSpaceRay.Direction.y,
            worldSpaceRay.Direction.z,
            t,
            face_index,
            hits,
            mesh_geometry->num_faces/*,
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

      const unsigned int v0_index = mesh_geometry->fv0[face_index];
      const unsigned int v1_index = mesh_geometry->fv1[face_index];
      const unsigned int v2_index = mesh_geometry->fv2[face_index];

      const Point v0 = Point(mesh_geometry->x_packed[v0_index], 
                             mesh_geometry->y_packed[v0_index],
                             mesh_geometry->z_packed[v0_index]);
      const Point v1 = Point(mesh_geometry->x_packed[v1_index],
                             mesh_geometry->y_packed[v1_index],
                             mesh_geometry->z_packed[v1_index]);
      const Point v2 = Point(mesh_geometry->x_packed[v2_index],
                             mesh_geometry->y_packed[v2_index],
                             mesh_geometry->z_packed[v2_index]);

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

      const unsigned int index = RandomUniformBetween(0u, mesh_geometry->num_faces - 1);
      return Point(mesh_geometry->x[index], mesh_geometry->y[index], mesh_geometry->z[index]);
   }

   void mesh_geometry::unpack_faces() {
      x.reserve(num_faces * 3);
      y.reserve(num_faces * 3);
      z.reserve(num_faces * 3);

      if (has_vertex_normals) {
         nx.reserve(num_faces * 3);
         ny.reserve(num_faces * 3);
         nz.reserve(num_faces * 3);
      }
      
      if (has_vertex_uvs) {
         u.reserve(num_faces * 3);
         v.reserve(num_faces * 3);
      }
      
      std::vector<std::vector<unsigned int>> fvs = { fv0, fv1, fv2 };
      
      for (const auto& fv : fvs) {
         for (unsigned int i = 0; i < num_faces; i++) {
            const unsigned int index = fv[i];

            x.push_back(x_packed[index]);
            y.push_back(y_packed[index]);
            z.push_back(z_packed[index]);

            if (has_vertex_normals) {
               nx.push_back(nx_packed[index]);
               ny.push_back(ny_packed[index]);
               nz.push_back(nz_packed[index]);
            }

            if (has_vertex_uvs) {
               u.push_back(u_packed[index]);
               v.push_back(v_packed[index]);
            }
         }
      }

      num_vertices = 3 * num_faces;
   }

   Point mesh_geometry::get_vertex(const unsigned int i) const {
      return {x_packed[i], y_packed[i], z_packed[i]};
   }

   void mesh_geometry::get_vertices_for_face(unsigned int i, poly::Point vertices[3]) const {
      vertices[0] = { x_packed[fv0[i]], y_packed[fv0[i]], z_packed[fv0[i]] };
      vertices[1] = { x_packed[fv1[i]], y_packed[fv1[i]], z_packed[fv1[i]] };
      vertices[2] = { x_packed[fv2[i]], y_packed[fv2[i]], z_packed[fv2[i]] };
   }
   
   Point3ui mesh_geometry::get_vertex_indices_for_face(const unsigned int i) const {
      return {fv0[i], fv1[i], fv2[i]};
   }

   Mesh::~Mesh() {

   }

   bool Mesh::hits(const Ray &world_ray, const unsigned int *face_indices, unsigned int num_face_indices) const {
      for (unsigned int face_index = 0; face_index < num_face_indices; face_index++) {

         Point v0 = {mesh_geometry->x_packed[mesh_geometry->fv0[face_index]],
                           mesh_geometry->y_packed[mesh_geometry->fv0[face_index]],
                           mesh_geometry->z_packed[mesh_geometry->fv0[face_index]]};
         Point v1 = {mesh_geometry->x_packed[mesh_geometry->fv1[face_index]],
                           mesh_geometry->y_packed[mesh_geometry->fv1[face_index]],
                           mesh_geometry->z_packed[mesh_geometry->fv1[face_index]]};
         Point v2 = {mesh_geometry->x_packed[mesh_geometry->fv2[face_index]],
                           mesh_geometry->y_packed[mesh_geometry->fv2[face_index]],
                           mesh_geometry->z_packed[mesh_geometry->fv2[face_index]]};

         object_to_world->ApplyInPlace(v0);
         object_to_world->ApplyInPlace(v1);
         object_to_world->ApplyInPlace(v2);
         
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

      Point v0_local = {mesh_geometry->x_packed[mesh_geometry->fv0[face_index]],
                        mesh_geometry->y_packed[mesh_geometry->fv0[face_index]],
                        mesh_geometry->z_packed[mesh_geometry->fv0[face_index]]};
      Point v1_local = {mesh_geometry->x_packed[mesh_geometry->fv1[face_index]],
                        mesh_geometry->y_packed[mesh_geometry->fv1[face_index]],
                        mesh_geometry->z_packed[mesh_geometry->fv1[face_index]]};
      Point v2_local = {mesh_geometry->x_packed[mesh_geometry->fv2[face_index]],
                        mesh_geometry->y_packed[mesh_geometry->fv2[face_index]],
                        mesh_geometry->z_packed[mesh_geometry->fv2[face_index]]};

      Point p0 = object_to_world->Apply(v0_local);
      Point p1 = object_to_world->Apply(v1_local);
      Point p2 = object_to_world->Apply(v2_local);
      
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
      
      bool debug = false;
      
      if (c - (a - b) < -.000001) {
         
         debug = true;
         ERROR("triangle side length fail :/\n");
      }

      const float sa = std::sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)));

      return sa;
   }
   
   void Mesh::recalculate_bounding_box() {
      
      // reset to min/impossible
      world_bb.p0 = { poly::Infinity, poly::Infinity, poly::Infinity};
      world_bb.p1 = { -poly::Infinity, -poly::Infinity, -poly::Infinity};
      
      // TODO PERF investigate performance of iterating over packed vs unpacked vertices
      for (int i = 0; i < mesh_geometry->num_vertices; i++) {
         const Point p = object_to_world->Apply(Point(mesh_geometry->x[i], mesh_geometry->y[i], mesh_geometry->z[i]));
         world_bb.UnionInPlace(p);
      }
   }
}
