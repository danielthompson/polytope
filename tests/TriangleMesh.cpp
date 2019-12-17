#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/Constants.h"
#include "../src/shapes/TriangleMesh.h"

namespace Tests {

   namespace TriangleMesh {

      TEST(TriangleMesh, Hits) {
         Polytope::Transform identity;
         Polytope::TriangleMesh tm(identity, nullptr);

         tm.Vertices.emplace_back(Polytope::Point(0, 0, 0));
         tm.Vertices.emplace_back(Polytope::Point(0, 1, 0));
         tm.Vertices.emplace_back(Polytope::Point(1, 0, 0));

         tm.Faces.emplace_back(Polytope::Point3ui(0, 1, 2));

         // hits, from either direction

         Polytope::Ray ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, 1));
         EXPECT_TRUE(tm.Hits(ray));

         ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, -1));
         EXPECT_TRUE(tm.Hits(ray));

         ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, 1));
         EXPECT_FALSE(tm.Hits(ray));

         ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, -1));
         EXPECT_FALSE(tm.Hits(ray));
      }

      TEST(TriangleMesh, Intersects) {
         Polytope::Transform identity;
         Polytope::TriangleMesh tm(identity, nullptr);

         tm.Vertices.emplace_back(Polytope::Point(0, 0, 0));
         tm.Vertices.emplace_back(Polytope::Point(0, 1, 0));
         tm.Vertices.emplace_back(Polytope::Point(1, 0, 0));

         tm.Faces.emplace_back(Polytope::Point3ui(0, 1, 2));

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
            EXPECT_EQ(Polytope::Point(0.2f, 0.2f, 0), intersection.Location);
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
      }
   }
}
