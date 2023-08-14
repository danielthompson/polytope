#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/constants.h"

namespace Tests {

   namespace Vector {
      using poly::vector;
      using poly::normal;

      namespace Equality {
         TEST(Vector, Equals) {
            vector p1 = vector(1.0f, 1.0f, 1.0f);
            vector p2 = vector(1.0f, 1.0f, 1.0f);

            EXPECT_EQ(p1, p1);
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p2, p2);
         }

         TEST(Vector, NotEquals) {
            vector p1 = vector(1.0f, 1.0f, 1.0f);
            vector p2 = vector(0.0f, 1.0f, 1.0f);
            EXPECT_NE(p1, p2);
         }
      }

      namespace AtOperator {
         TEST(Vector, At1) {
            vector v = vector(1.0f, 2.0f, 3.0f);

            EXPECT_EQ(1.0f, v[0]);
            EXPECT_EQ(2.0f, v[1]);
            EXPECT_EQ(3.0f, v[2]);
         }
      }

      namespace UnaryMinusOperator {
         TEST(Vector, UnaryMinusOperator1) {
            vector v0 = vector(1.0f, 2.0f, 3.0f);
            vector actual = -v0;

            EXPECT_NE(v0, actual);

            EXPECT_EQ(-1.0f, actual.x);
            EXPECT_EQ(-2.0f, actual.y);
            EXPECT_EQ(-3.0f, actual.z);
         }
      }

      namespace PlusOperator {
         TEST(Vector, PlusOperator1) {
            vector v0 = vector(1.0f, 2.0f, 3.0f);
            vector v1 = vector(4.0f, 5.0f, 6.0f);
            vector actual = v0 + v1;

            EXPECT_NE(v0, actual);
            EXPECT_NE(v1, actual);

            EXPECT_EQ(5, actual.x);
            EXPECT_EQ(7, actual.y);
            EXPECT_EQ(9, actual.z);
         }
      }

      namespace CrossNormal {
         TEST(Vector, CrossNormal1) {
            vector p1 = vector(1, 2, 3);
            normal p2 = normal(4, 5, 6);

            vector actual = p1.cross(p2);
            vector expected = vector(-3, 6, -3);

            EXPECT_EQ(actual, expected);
         }
      }

      namespace Dot {
         TEST(Vector, Dot1) {
            vector p1 = vector(1, 2, 3);
            vector p2 = vector(4, 5, 6);

            float actual = p1.dot(p2);
            float expected = 32.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Vector, Dot2) {
            vector p1 = vector(-1, 2, 3);
            vector p2 = vector(4, 5, 6);

            float actual = p1.dot(p2);
            float expected = 24.0f;

            EXPECT_EQ(actual, expected);
         }

         TEST(Vector, Dot3) {
            vector p1 = vector(0, 0, 0);
            vector p2 = vector(0, 0, 0);

            float actual = p1.dot(p2);
            float expected = 0.f;

            EXPECT_EQ(actual, expected);
         }
      }
   }
}