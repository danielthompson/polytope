#include "gtest/gtest.h"

#include "../../src/cpu/structures/Vectors.h"
#include "../../src/cpu/constants.h"
#include "../../src/cpu/structures/Intersection.h"
#include "../../src/cpu/shapes/linear_soa/mesh_linear_soa.h"

namespace Tests {
   TEST(MeshLinearSOA, Intersects) {
      std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
      Polytope::MeshLinearSOA tm(identity, identity, nullptr);

      // face 0
      
      tm.x_packed.push_back(0);
      tm.y_packed.push_back(0);
      tm.z_packed.push_back(0);

      tm.x_packed.push_back(0);
      tm.y_packed.push_back(1);
      tm.z_packed.push_back(0);

      tm.x_packed.push_back(1);
      tm.y_packed.push_back(0);
      tm.z_packed.push_back(0);

      tm.fv0.push_back(0);
      tm.fv1.push_back(1);
      tm.fv2.push_back(2);

      // face 1

      tm.x_packed.push_back(0);
      tm.y_packed.push_back(0);
      tm.z_packed.push_back(1);

      tm.x_packed.push_back(0);
      tm.y_packed.push_back(1);
      tm.z_packed.push_back(1);

      tm.x_packed.push_back(1);
      tm.y_packed.push_back(0);
      tm.z_packed.push_back(1);

      tm.fv0.push_back(3);
      tm.fv1.push_back(4);
      tm.fv2.push_back(5);

      tm.num_vertices_packed = 6;
      tm.num_faces = 2;
      
      tm.unpack_faces();
      
      // hits, from either direction

      {
         Polytope::Ray ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, 1));
         Polytope::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_TRUE(intersection.Hits);
         EXPECT_EQ(&tm, intersection.Shape);
         EXPECT_EQ(Polytope::Normal(0, 0, -1), intersection.Normal);
         EXPECT_EQ(Polytope::Point(0.2f, 0.2f, 0), intersection.Location);
      }

      {
         Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, -1));
         Polytope::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_TRUE(intersection.Hits);
         EXPECT_EQ(&tm, intersection.Shape);
         EXPECT_EQ(Polytope::Normal(0, 0, 1), intersection.Normal);
         EXPECT_EQ(Polytope::Point(0.2f, 0.2f, 1), intersection.Location);
      }

      {
         Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, 1));
         Polytope::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      {
         Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, -1));
         Polytope::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      // parallel
      {
         Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 0.5f), Polytope::Vector(0, 1, 0));
         Polytope::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }
   }

}
