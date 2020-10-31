#include "gtest/gtest.h"

#include "../../src/cpu/structures/Vectors.h"
#include "../../src/cpu/constants.h"
#include "../../src/cpu/structures/Intersection.h"
#include "../../src/cpu/shapes/mesh.h"

namespace Tests {
   TEST(Mesh, Intersects) {
      std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
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
         poly::Ray ray(poly::Point(0.2f, 0.2f, -10), poly::Vector(0, 0, 1));
         poly::Intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_TRUE(intersection.Hits);
         EXPECT_EQ(&tm, intersection.Shape);
         EXPECT_EQ(poly::Normal(0, 0, -1), intersection.geo_normal);
         EXPECT_EQ(poly::Point(0.2f, 0.2f, 0), intersection.Location);
      }

      {
         poly::Ray ray = poly::Ray(poly::Point(0.2f, 0.2f, 10), poly::Vector(0, 0, -1));
         poly::Intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_TRUE(intersection.Hits);
         EXPECT_EQ(&tm, intersection.Shape);
         EXPECT_EQ(poly::Normal(0, 0, 1), intersection.geo_normal);
         EXPECT_EQ(poly::Point(0.2f, 0.2f, 1), intersection.Location);
      }

      {
         poly::Ray ray = poly::Ray(poly::Point(0.2f, 0.2f, 10), poly::Vector(0, 0, 1));
         poly::Intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      {
         poly::Ray ray = poly::Ray(poly::Point(0.2f, 0.2f, -10), poly::Vector(0, 0, -1));
         poly::Intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      // parallel
      {
         poly::Ray ray = poly::Ray(poly::Point(0.2f, 0.2f, 0.5f), poly::Vector(0, 1, 0));
         poly::Intersection intersection;
         tm.intersect(ray, intersection);
         EXPECT_FALSE(intersection.Hits);
      }
   }

}
