#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/shapes/AbstractShape.h"

namespace Tests {

   namespace BoundingBox {
      using Polytope::Point;
      using Polytope::Vector;
      using Polytope::Ray;
      using Polytope::Intersection;
      using Polytope::BoundingBox;

      namespace Intersect {
         TEST(BoundingBox, Intersect) {
            Point min(0.0f, 0.0f, 0.0f);
            Point max(1.0f, 1.0f, 1.0f);

            BoundingBox b(min, max);

            Point origin(0.5, 0.5, 10);
            Vector direction(0, 0, -1);

            Ray ray(origin, direction);

            Intersection intersection;
            // TODO fix
//            b.Intersect(ray, &intersection);
//
//            EXPECT_TRUE(intersection.Hits);
//            EXPECT_EQ(ray.MinT, 9);
         }
      }

      namespace Union {
         TEST(BoundingBox, UnionSelf) {
            Point min(0.0f, 0.0f, 0.0f);
            Point max(1.0f, 1.0f, 1.0f);

            BoundingBox b(min, max);

            BoundingBox actual = b.Union(b);

            EXPECT_EQ(actual.p0.x, min.x);
            EXPECT_EQ(actual.p0.y, min.y);
            EXPECT_EQ(actual.p0.z, min.z);

            EXPECT_EQ(actual.p1.x, max.x);
            EXPECT_EQ(actual.p1.y, max.y);
            EXPECT_EQ(actual.p1.z, max.z);
         }

         TEST(BoundingBox, Union) {
            Point min(0.0f, 0.0f, 0.0f);
            Point max(1.0f, 1.0f, 1.0f);

            BoundingBox b(min, max);

            Point othermin(0.25f, 0.25f, 0.0f);
            Point othermax(1.25f, 0.5f, 1.0f);

            BoundingBox other(othermin, othermax);

            BoundingBox actual = b.Union(other);

            EXPECT_EQ(actual.p0.x, 0);
            EXPECT_EQ(actual.p0.y, 0);
            EXPECT_EQ(actual.p0.z, 0);

            EXPECT_EQ(actual.p1.x, 1.25);
            EXPECT_EQ(actual.p1.y, 1);
            EXPECT_EQ(actual.p1.z, 1);
         }
      }
   }
}