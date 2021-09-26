#include "gtest/gtest.h"

#include "../../src/cpu/structures/Vectors.h"
#include "../../src/cpu/constants.h"
#include "../../src/cpu/structures/intersection.h"
#include "../../src/cpu/shapes/mesh.h"

namespace Tests {
   TEST(mesh_linear_ispc, intersects) {
      std::shared_ptr<poly::transform> identity = std::make_shared<poly::transform>();
      std::shared_ptr<poly::mesh_geometry> geometry = std::make_shared<poly::mesh_geometry>(); 
      poly::Mesh tm(identity, identity, nullptr, geometry);

      // face 0
      
      geometry->x_packed.push_back(0);
      geometry->y_packed.push_back(0);
      geometry->z_packed.push_back(0);

      geometry->x_packed.push_back(0);
      geometry->y_packed.push_back(1);
      geometry->z_packed.push_back(0);

      geometry->x_packed.push_back(1);
      geometry->y_packed.push_back(0);
      geometry->z_packed.push_back(0);

      geometry->fv0.push_back(0);
      geometry->fv1.push_back(1);
      geometry->fv2.push_back(2);

      // face 1

      geometry->x_packed.push_back(0);
      geometry->y_packed.push_back(0);
      geometry->z_packed.push_back(1);

      geometry->x_packed.push_back(0);
      geometry->y_packed.push_back(1);
      geometry->z_packed.push_back(1);

      geometry->x_packed.push_back(1);
      geometry->y_packed.push_back(0);
      geometry->z_packed.push_back(1);

      geometry->fv0.push_back(3);
      geometry->fv1.push_back(4);
      geometry->fv2.push_back(5);

      geometry->num_vertices_packed = 6;
      geometry->num_faces = 2;
      
      geometry->unpack_faces();
      
      // hits, from either direction

      {
         poly::ray ray(poly::point(0.2f, 0.2f, -10), poly::vector(0, 0, 1));
         poly::intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_TRUE(intersection.Hits);
         EXPECT_EQ(&tm, intersection.shape);
         EXPECT_EQ(poly::normal(0, 0, -1), intersection.geo_normal);
         EXPECT_EQ(poly::point(0.2f, 0.2f, 0), intersection.location);
      }

      {
         poly::ray ray = poly::ray(poly::point(0.2f, 0.2f, 10), poly::vector(0, 0, -1));
         poly::intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_TRUE(intersection.Hits);
         EXPECT_EQ(&tm, intersection.shape);
         EXPECT_EQ(poly::normal(0, 0, 1), intersection.geo_normal);
         EXPECT_EQ(poly::point(0.2f, 0.2f, 1), intersection.location);
      }

      {
         poly::ray ray = poly::ray(poly::point(0.2f, 0.2f, 10), poly::vector(0, 0, 1));
         poly::intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      {
         poly::ray ray = poly::ray(poly::point(0.2f, 0.2f, -10), poly::vector(0, 0, -1));
         poly::intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      // parallel
      {
         poly::ray ray = poly::ray(poly::point(0.2f, 0.2f, 0.5f), poly::vector(0, 1, 0));
         poly::intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_FALSE(intersection.Hits);
      }
   }

}
