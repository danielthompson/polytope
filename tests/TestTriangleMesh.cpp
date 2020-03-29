#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/Constants.h"
#include "../src/shapes/triangle.h"
#include "../src/structures/Intersection.h"

namespace Tests {

   namespace TriangleMesh {

      TEST(TriangleMesh, Hits) {
         std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::TriangleMesh tm(identity, identity, nullptr);

         tm.Vertices.emplace_back(Polytope::Point(0, 0, 0));
         tm.Vertices.emplace_back(Polytope::Point(0, 1, 0));
         tm.Vertices.emplace_back(Polytope::Point(1, 0, 0));

         tm.Faces.emplace_back(Polytope::Point3ui(0, 1, 2));

         // hits, from either direction

         {
            Polytope::Intersection intersection;
            Polytope::Ray ray(0.2f, 0.2f, -10, 0, 0, 1);
            tm.Intersect(ray, &intersection);
            EXPECT_TRUE(intersection.Hits);
         }

         {
            Polytope::Intersection intersection;
            Polytope::Ray ray(0.2f, 0.2f, 10, 0, 0, -1);
            tm.Intersect(ray, &intersection);
            EXPECT_TRUE(intersection.Hits);
         }

         {
            Polytope::Intersection intersection;
            Polytope::Ray ray(0.2f, 0.2f, 10, 0, 0, 1);
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }

         {
            Polytope::Intersection intersection;
            Polytope::Ray ray(0.2f, 0.2f, -10, 0, 0, -1);
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }

         {
            Polytope::Intersection intersection;
            Polytope::Ray ray(0.2f, 0.2f, -10, 0, 1, 0);
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }
      }

      TEST(TriangleMesh, Intersects) {
         std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::TriangleMesh tm(identity, identity, nullptr);

         tm.Vertices.emplace_back(Polytope::Point(0, 0, 0));
         tm.Vertices.emplace_back(Polytope::Point(0, 1, 0));
         tm.Vertices.emplace_back(Polytope::Point(1, 0, 0));

         tm.Vertices.emplace_back(Polytope::Point(0, 0, 1));
         tm.Vertices.emplace_back(Polytope::Point(0, 1, 1));
         tm.Vertices.emplace_back(Polytope::Point(1, 0, 1));

         tm.Faces.emplace_back(Polytope::Point3ui(0, 1, 2));
         tm.Faces.emplace_back(Polytope::Point3ui(3, 4, 5));

         // hits, from either direction

         {
            Polytope::Ray ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, 1));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_TRUE(intersection.Hits);
            EXPECT_EQ(&tm, intersection.Shape);
            EXPECT_EQ(Polytope::Normal(0, 0, -1), intersection.Normal);
            EXPECT_EQ(Polytope::Point(0.2f, 0.2f, 0), intersection.Location);
         }

         {
            Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, -1));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_TRUE(intersection.Hits);
            EXPECT_EQ(&tm, intersection.Shape);
            EXPECT_EQ(Polytope::Normal(0, 0, 1), intersection.Normal);
            EXPECT_EQ(Polytope::Point(0.2f, 0.2f, 1), intersection.Location);
         }

         {
            Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, 1));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }

         {
            Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, -1));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }

         // parallel
         {
            Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 0.5f), Polytope::Vector(0, 1, 0));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }
      }
   }
}
