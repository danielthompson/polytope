#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/bounding_box.h"


namespace Tests {
   namespace BoundingBox {
      namespace Hit {
         TEST(BoundingBox, Hits1) {
            poly::point min(0.0f, 0.0f, 0.0f);
            poly::point max(1.0f, 1.0f, 1.0f);

            poly::bounding_box b(min, max);

            poly::point origin(0.5, 0.5, 10);
            poly::vector direction(0, 0, -1);

            poly::ray ray(origin, direction);

            const poly::vector inverse_direction = {
                  1.f / ray.direction.x,
                  1.f / ray.direction.y,
                  1.f / ray.direction.z
            };
            
            bool actual = b.hits(ray, inverse_direction);

            EXPECT_TRUE(actual);
         }
      }

      namespace Union {
         TEST(BoundingBox, UnionSelf) {
            poly::point min(0.0f, 0.0f, 0.0f);
            poly::point max(1.0f, 1.0f, 1.0f);

            poly::bounding_box b(min, max);

            poly::bounding_box actual = b.Union(b);

            EXPECT_EQ(actual.p0.x, min.x);
            EXPECT_EQ(actual.p0.y, min.y);
            EXPECT_EQ(actual.p0.z, min.z);

            EXPECT_EQ(actual.p1.x, max.x);
            EXPECT_EQ(actual.p1.y, max.y);
            EXPECT_EQ(actual.p1.z, max.z);
         }

         TEST(BoundingBox, UnionWithBox) {
            poly::point min(0.0f, 0.0f, 0.0f);
            poly::point max(1.0f, 1.0f, 1.0f);

            poly::bounding_box b(min, max);

            poly::point othermin(0.25f, 0.25f, 0.0f);
            poly::point othermax(1.25f, 0.5f, 1.0f);

            poly::bounding_box other(othermin, othermax);

            poly::bounding_box actual = b.Union(other);

            EXPECT_EQ(actual.p0.x, 0);
            EXPECT_EQ(actual.p0.y, 0);
            EXPECT_EQ(actual.p0.z, 0);

            EXPECT_EQ(actual.p1.x, 1.25);
            EXPECT_EQ(actual.p1.y, 1);
            EXPECT_EQ(actual.p1.z, 1);
         }

         TEST(BoundingBox, UnionWithPoint1) {
            poly::point min(0.0f, 0.0f, 0.0f);
            poly::point max(1.0f, 1.0f, 1.0f);

            poly::bounding_box b(min, max);

            poly::point p(0.25f, 0.25f, 0.25f);

            poly::bounding_box actual = b.Union(p);

            EXPECT_EQ(actual.p0.x, 0);
            EXPECT_EQ(actual.p0.y, 0);
            EXPECT_EQ(actual.p0.z, 0);

            EXPECT_EQ(actual.p1.x, 1);
            EXPECT_EQ(actual.p1.y, 1);
            EXPECT_EQ(actual.p1.z, 1);
         }

         TEST(BoundingBox, UnionWithPoint2) {
            poly::point min(0.0f, 0.0f, 0.0f);
            poly::point max(1.0f, 1.0f, 1.0f);

            poly::bounding_box b(min, max);

            poly::point p(1.25f, 1.5f, 1.75f);

            poly::bounding_box actual = b.Union(p);

            EXPECT_EQ(actual.p0.x, 0);
            EXPECT_EQ(actual.p0.y, 0);
            EXPECT_EQ(actual.p0.z, 0);

            EXPECT_EQ(actual.p1.x, 1.25);
            EXPECT_EQ(actual.p1.y, 1.5);
            EXPECT_EQ(actual.p1.z, 1.75);
         }

         TEST(BoundingBox, UnionWithPoint3) {
            poly::point min(0.0f, 0.0f, 0.0f);
            poly::point max(1.0f, 1.0f, 1.0f);

            poly::bounding_box b(min, max);

            poly::point p(-1.25f, -1.5f, -1.75f);

            poly::bounding_box actual = b.Union(p);

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
