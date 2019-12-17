#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"

namespace Tests {

   namespace Point {
      using Polytope::Point;
      using Polytope::Vector;

      namespace Equality {
         TEST(Equality, Equals) {
            Point p1 = Point(1.0f, 1.0f, 1.0f);
            Point p2 = Point(1.0f, 1.0f, 1.0f);

            EXPECT_EQ(p1, p1);
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p2, p2);
         }

         TEST(Equality, NotEquals) {
            Point p1 = Point(1.0f, 1.0f, 1.0f);
            Point p2 = Point(0.0f, 1.0f, 1.0f);
            EXPECT_NE(p1, p2);
         }
      }

      namespace AtOperator {
         TEST(Equality, At1) {
            Point p = Point(1.0f, 2.0f, 3.0f);

            EXPECT_EQ(1.0f, p[0]);
            EXPECT_EQ(2.0f, p[1]);
            EXPECT_EQ(3.0f, p[2]);
         }
      }

      namespace Dot {
         TEST(Dot, Dot1) {
            Point p1 = Point(1, 2, 3);
            Point p2 = Point(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 32.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Dot, DotVector1) {
            Point p1 = Point(1, 2, 3);
            Vector p2 = Vector(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 32.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Dot, Dot2) {
            Point p1 = Point(-1, 2, 3);
            Point p2 = Point(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 24.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Dot, DotVector2) {
            Point p1 = Point(-1, 2, 3);
            Vector p2 = Vector(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 24.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Dot, Dot3) {
            Point p1 = Point(0, 0, 0);
            Point p2 = Point(0, 0, 0);

            float actual = p1.Dot(p2);
            float expected = 0.f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Dot, DotVector3) {
            Point p1 = Point(0, 0, 0);
            Vector p2 = Vector(0, 0, 0);

            float actual = p1.Dot(p2);
            float expected = 0.f;

            EXPECT_EQ(actual, expected);
         }
      }
   }
}
