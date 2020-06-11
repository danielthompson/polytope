#include "gtest/gtest.h"

#include "../../src/cpu/structures/Vectors.h"
#include "../../src/cpu/constants.h"

#include "../../src/cpu/structures/Intersection.h"
#include "../../src/cpu/shapes/linear_aos/mesh_linear_aos.h"

namespace Tests {
   TEST(MeshLinearAOS, Hits) {
      std::shared_ptr<poly::Transform> identity;
      identity = std::make_shared<poly::Transform>();
      poly::MeshLinearAOS tm(identity, identity, nullptr);

      tm.Vertices.emplace_back(poly::Point(0, 0, 0));
      tm.Vertices.emplace_back(poly::Point(0, 1, 0));
      tm.Vertices.emplace_back(poly::Point(1, 0, 0));

      tm.Faces.emplace_back(poly::Point3ui(0, 1, 2));

      // hits, from either direction

      {
         poly::Intersection intersection;
         poly::Ray ray(0.2f, 0.2f, -10, 0, 0, 1);
         tm.intersect(ray, &intersection);
         EXPECT_TRUE(intersection.Hits);
      }

      {
         poly::Intersection intersection;
         poly::Ray ray(0.2f, 0.2f, 10, 0, 0, -1);
         tm.intersect(ray, &intersection);
         EXPECT_TRUE(intersection.Hits);
      }

      {
         poly::Intersection intersection;
         poly::Ray ray(0.2f, 0.2f, 10, 0, 0, 1);
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      {
         poly::Intersection intersection;
         poly::Ray ray(0.2f, 0.2f, -10, 0, 0, -1);
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      {
         poly::Intersection intersection;
         poly::Ray ray(0.2f, 0.2f, -10, 0, 1, 0);
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }
   }

   TEST(MeshLinearAOS, Intersects) {
      std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
      poly::MeshLinearAOS tm(identity, identity, nullptr);

      tm.Vertices.emplace_back(poly::Point(0, 0, 0));
      tm.Vertices.emplace_back(poly::Point(0, 1, 0));
      tm.Vertices.emplace_back(poly::Point(1, 0, 0));

      tm.Vertices.emplace_back(poly::Point(0, 0, 1));
      tm.Vertices.emplace_back(poly::Point(0, 1, 1));
      tm.Vertices.emplace_back(poly::Point(1, 0, 1));

      tm.Faces.emplace_back(poly::Point3ui(0, 1, 2));
      tm.Faces.emplace_back(poly::Point3ui(3, 4, 5));

      // hits, from either direction

      {
         poly::Ray ray(poly::Point(0.2f, 0.2f, -10), poly::Vector(0, 0, 1));
         poly::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_TRUE(intersection.Hits);
         EXPECT_EQ(&tm, intersection.Shape);
         EXPECT_EQ(poly::Normal(0, 0, -1), intersection.Normal);
         EXPECT_EQ(poly::Point(0.2f, 0.2f, 0), intersection.Location);
      }

      {
         poly::Ray ray = poly::Ray(poly::Point(0.2f, 0.2f, 10), poly::Vector(0, 0, -1));
         poly::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_TRUE(intersection.Hits);
         EXPECT_EQ(&tm, intersection.Shape);
         EXPECT_EQ(poly::Normal(0, 0, 1), intersection.Normal);
         EXPECT_EQ(poly::Point(0.2f, 0.2f, 1), intersection.Location);
      }

      {
         poly::Ray ray = poly::Ray(poly::Point(0.2f, 0.2f, 10), poly::Vector(0, 0, 1));
         poly::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      {
         poly::Ray ray = poly::Ray(poly::Point(0.2f, 0.2f, -10), poly::Vector(0, 0, -1));
         poly::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }

      // parallel
      {
         poly::Ray ray = poly::Ray(poly::Point(0.2f, 0.2f, 0.5f), poly::Vector(0, 1, 0));
         poly::Intersection intersection;
         tm.intersect(ray, &intersection);
         EXPECT_FALSE(intersection.Hits);
      }
   }
}
