#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"

namespace Tests {

   namespace Point {
      using poly::normal;
      using poly::point;
      using poly::vector;

      namespace Equality {
         TEST(Point, Equals) {
            point p1 = point(1.0f, 1.0f, 1.0f);
            point p2 = point(1.0f, 1.0f, 1.0f);

            EXPECT_EQ(p1, p1);
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p2, p2);
         }

         TEST(Point, NotEquals) {
            point p1 = point(1.0f, 1.0f, 1.0f);
            point p2 = point(0.0f, 1.0f, 1.0f);
            EXPECT_NE(p1, p2);
         }
      }

      namespace AtOperator {
         TEST(Point, AtOperator1) {
            point p = point(1.0f, 2.0f, 3.0f);

            EXPECT_EQ(1.0f, p[0]);
            EXPECT_EQ(2.0f, p[1]);
            EXPECT_EQ(3.0f, p[2]);
         }
      }

      namespace PlusOperator {
         TEST(Point, PlusOperator1) {
            point p1 = point(1.0f, 2.0f, 3.0f);
            point p2 = point(4.0f, 5.0f, 6.0f);

            point actual = p1 + p2;

            EXPECT_EQ(5.0f, actual.x);
            EXPECT_EQ(7.0f, actual.y);
            EXPECT_EQ(9.0f, actual.z);
         }
      }

      namespace PlusEqualsOperator {
         TEST(Point, PlusEqualsOperatorVector1) {
            point p = point(1.0f, 2.0f, 3.0f);
            vector v = vector(4.0f, 5.0f, 6.0f);

            p += v;

            EXPECT_EQ(5.0f, p.x);
            EXPECT_EQ(7.0f, p.y);
            EXPECT_EQ(9.0f, p.z);
         }

         TEST(Point, PlusEqualsOperatorNormal1) {
            point p = point(1.0f, 2.0f, 3.0f);
            normal n = normal(4.0f, 5.0f, 6.0f);

            p += n;

            EXPECT_EQ(5.0f, p.x);
            EXPECT_EQ(7.0f, p.y);
            EXPECT_EQ(9.0f, p.z);
         }
      }

      namespace Dot {
         TEST(Point, Dot1) {
            point p1 = point(1, 2, 3);
            point p2 = point(4, 5, 6);

            float actual = p1.dot(p2);
            float expected = 32.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, DotVector1) {
            point p1 = point(1, 2, 3);
            vector p2 = vector(4, 5, 6);

            float actual = p1.dot(p2);
            float expected = 32.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, Dot2) {
            point p1 = point(-1, 2, 3);
            point p2 = point(4, 5, 6);

            float actual = p1.dot(p2);
            float expected = 24.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, DotVector2) {
            point p1 = point(-1, 2, 3);
            vector p2 = vector(4, 5, 6);

            float actual = p1.dot(p2);
            float expected = 24.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, Dot3) {
            point p1 = point(0, 0, 0);
            point p2 = point(0, 0, 0);

            float actual = p1.dot(p2);
            float expected = 0.f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, DotVector3) {
            point p1 = point(0, 0, 0);
            vector p2 = vector(0, 0, 0);

            float actual = p1.dot(p2);
            float expected = 0.f;

            EXPECT_EQ(actual, expected);
         }
      }
   }
}
