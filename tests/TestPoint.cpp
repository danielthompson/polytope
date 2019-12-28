#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"

namespace Tests {

   namespace Point {
      using Polytope::Normal;
      using Polytope::Point;
      using Polytope::Vector;

      namespace Equality {
         TEST(Point, Equals) {
            Point p1 = Point(1.0f, 1.0f, 1.0f);
            Point p2 = Point(1.0f, 1.0f, 1.0f);

            EXPECT_EQ(p1, p1);
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p2, p2);
         }

         TEST(Point, NotEquals) {
            Point p1 = Point(1.0f, 1.0f, 1.0f);
            Point p2 = Point(0.0f, 1.0f, 1.0f);
            EXPECT_NE(p1, p2);
         }
      }

      namespace AtOperator {
         TEST(Point, AtOperator1) {
            Point p = Point(1.0f, 2.0f, 3.0f);

            EXPECT_EQ(1.0f, p[0]);
            EXPECT_EQ(2.0f, p[1]);
            EXPECT_EQ(3.0f, p[2]);
         }
      }

      namespace PlusOperator {
         TEST(Point, PlusOperator1) {
            Point p1 = Point(1.0f, 2.0f, 3.0f);
            Point p2 = Point(4.0f, 5.0f, 6.0f);

            Point actual = p1 + p2;

            EXPECT_EQ(5.0f, actual.x);
            EXPECT_EQ(7.0f, actual.y);
            EXPECT_EQ(9.0f, actual.z);
         }
      }

      namespace PlusEqualsOperator {
         TEST(Point, PlusEqualsOperatorVector1) {
            Point p = Point(1.0f, 2.0f, 3.0f);
            Vector v = Vector(4.0f, 5.0f, 6.0f);

            p += v;

            EXPECT_EQ(5.0f, p.x);
            EXPECT_EQ(7.0f, p.y);
            EXPECT_EQ(9.0f, p.z);
         }

         TEST(Point, PlusEqualsOperatorNormal1) {
            Point p = Point(1.0f, 2.0f, 3.0f);
            Normal n = Normal(4.0f, 5.0f, 6.0f);

            p += n;

            EXPECT_EQ(5.0f, p.x);
            EXPECT_EQ(7.0f, p.y);
            EXPECT_EQ(9.0f, p.z);
         }
      }

      namespace Dot {
         TEST(Point, Dot1) {
            Point p1 = Point(1, 2, 3);
            Point p2 = Point(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 32.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, DotVector1) {
            Point p1 = Point(1, 2, 3);
            Vector p2 = Vector(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 32.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, Dot2) {
            Point p1 = Point(-1, 2, 3);
            Point p2 = Point(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 24.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, DotVector2) {
            Point p1 = Point(-1, 2, 3);
            Vector p2 = Vector(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 24.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, Dot3) {
            Point p1 = Point(0, 0, 0);
            Point p2 = Point(0, 0, 0);

            float actual = p1.Dot(p2);
            float expected = 0.f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Point, DotVector3) {
            Point p1 = Point(0, 0, 0);
            Vector p2 = Vector(0, 0, 0);

            float actual = p1.Dot(p2);
            float expected = 0.f;

            EXPECT_EQ(actual, expected);
         }
      }
   }
}
