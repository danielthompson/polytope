#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/constants.h"

namespace Tests {

   namespace Vector {
      using poly::Vector;
      using poly::Normal;

      namespace Equality {
         TEST(Vector, Equals) {
            Vector p1 = Vector(1.0f, 1.0f, 1.0f);
            Vector p2 = Vector(1.0f, 1.0f, 1.0f);

            EXPECT_EQ(p1, p1);
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p2, p2);
         }

         TEST(Vector, NotEquals) {
            Vector p1 = Vector(1.0f, 1.0f, 1.0f);
            Vector p2 = Vector(0.0f, 1.0f, 1.0f);
            EXPECT_NE(p1, p2);
         }
      }

      namespace AtOperator {
         TEST(Vector, At1) {
            Vector v = Vector(1.0f, 2.0f, 3.0f);

            EXPECT_EQ(1.0f, v[0]);
            EXPECT_EQ(2.0f, v[1]);
            EXPECT_EQ(3.0f, v[2]);
         }
      }

      namespace UnaryMinusOperator {
         TEST(Vector, UnaryMinusOperator1) {
            Vector v0 = Vector(1.0f, 2.0f, 3.0f);
            Vector actual = -v0;

            EXPECT_NE(v0, actual);

            EXPECT_EQ(-1.0f, actual.x);
            EXPECT_EQ(-2.0f, actual.y);
            EXPECT_EQ(-3.0f, actual.z);
         }
      }

      namespace PlusOperator {
         TEST(Vector, PlusOperator1) {
            Vector v0 = Vector(1.0f, 2.0f, 3.0f);
            Vector v1 = Vector(4.0f, 5.0f, 6.0f);
            Vector actual = v0 + v1;

            EXPECT_NE(v0, actual);
            EXPECT_NE(v1, actual);

            EXPECT_EQ(5, actual.x);
            EXPECT_EQ(7, actual.y);
            EXPECT_EQ(9, actual.z);
         }
      }

      namespace CrossNormal {
         TEST(Vector, CrossNormal1) {
            Vector p1 = Vector(1, 2, 3);
            Normal p2 = Normal(4, 5, 6);

            Vector actual = p1.Cross(p2);
            Vector expected = Vector(-3, 6, -3);

            EXPECT_EQ(actual, expected);
         }
      }

      namespace Dot {
         TEST(Vector, Dot1) {
            Vector p1 = Vector(1, 2, 3);
            Vector p2 = Vector(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 32.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Vector, Dot2) {
            Vector p1 = Vector(-1, 2, 3);
            Vector p2 = Vector(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 24.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Vector, Dot3) {
            Vector p1 = Vector(0, 0, 0);
            Vector p2 = Vector(0, 0, 0);

            float actual = p1.Dot(p2);
            float expected = 0.f;

            EXPECT_EQ(actual, expected);
         }
      }
   }
}