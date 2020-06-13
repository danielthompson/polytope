#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/BoundingBox.h"


namespace Tests {
   namespace BoundingBox {
      namespace Hit {
         TEST(BoundingBox, Hits1) {
            poly::Point min(0.0f, 0.0f, 0.0f);
            poly::Point max(1.0f, 1.0f, 1.0f);

            poly::BoundingBox b(min, max);

            poly::Point origin(0.5, 0.5, 10);
            poly::Vector direction(0, 0, -1);

            poly::Ray ray(origin, direction);

            bool actual = b.Hits(ray);

            EXPECT_TRUE(actual);
         }
      }

      namespace Union {
         TEST(BoundingBox, UnionSelf) {
            poly::Point min(0.0f, 0.0f, 0.0f);
            poly::Point max(1.0f, 1.0f, 1.0f);

            poly::BoundingBox b(min, max);

            poly::BoundingBox actual = b.Union(b);

            EXPECT_EQ(actual.p0.x, min.x);
            EXPECT_EQ(actual.p0.y, min.y);
            EXPECT_EQ(actual.p0.z, min.z);

            EXPECT_EQ(actual.p1.x, max.x);
            EXPECT_EQ(actual.p1.y, max.y);
            EXPECT_EQ(actual.p1.z, max.z);
         }

         TEST(BoundingBox, UnionWithBox) {
            poly::Point min(0.0f, 0.0f, 0.0f);
            poly::Point max(1.0f, 1.0f, 1.0f);

            poly::BoundingBox b(min, max);

            poly::Point othermin(0.25f, 0.25f, 0.0f);
            poly::Point othermax(1.25f, 0.5f, 1.0f);

            poly::BoundingBox other(othermin, othermax);

            poly::BoundingBox actual = b.Union(other);

            EXPECT_EQ(actual.p0.x, 0);
            EXPECT_EQ(actual.p0.y, 0);
            EXPECT_EQ(actual.p0.z, 0);

            EXPECT_EQ(actual.p1.x, 1.25);
            EXPECT_EQ(actual.p1.y, 1);
            EXPECT_EQ(actual.p1.z, 1);
         }

         TEST(BoundingBox, UnionWithPoint1) {
            poly::Point min(0.0f, 0.0f, 0.0f);
            poly::Point max(1.0f, 1.0f, 1.0f);

            poly::BoundingBox b(min, max);

            poly::Point p(0.25f, 0.25f, 0.25f);

            poly::BoundingBox actual = b.Union(p);

            EXPECT_EQ(actual.p0.x, 0);
            EXPECT_EQ(actual.p0.y, 0);
            EXPECT_EQ(actual.p0.z, 0);

            EXPECT_EQ(actual.p1.x, 1);
            EXPECT_EQ(actual.p1.y, 1);
            EXPECT_EQ(actual.p1.z, 1);
         }

         TEST(BoundingBox, UnionWithPoint2) {
            poly::Point min(0.0f, 0.0f, 0.0f);
            poly::Point max(1.0f, 1.0f, 1.0f);

            poly::BoundingBox b(min, max);

            poly::Point p(1.25f, 1.5f, 1.75f);

            poly::BoundingBox actual = b.Union(p);

            EXPECT_EQ(actual.p0.x, 0);
            EXPECT_EQ(actual.p0.y, 0);
            EXPECT_EQ(actual.p0.z, 0);

            EXPECT_EQ(actual.p1.x, 1.25);
            EXPECT_EQ(actual.p1.y, 1.5);
            EXPECT_EQ(actual.p1.z, 1.75);
         }

         TEST(BoundingBox, UnionWithPoint3) {
            poly::Point min(0.0f, 0.0f, 0.0f);
            poly::Point max(1.0f, 1.0f, 1.0f);

            poly::BoundingBox b(min, max);

            poly::Point p(-1.25f, -1.5f, -1.75f);

            poly::BoundingBox actual = b.Union(p);

            EXPECT_EQ(actual.p0.x, -1.25);
            EXPECT_EQ(actual.p0.y, -1.5);
            EXPECT_EQ(actual.p0.z, -1.75);

            EXPECT_EQ(actual.p1.x, 1);
            EXPECT_EQ(actual.p1.y, 1);
            EXPECT_EQ(actual.p1.z, 1);
         }
      }
   }
}
