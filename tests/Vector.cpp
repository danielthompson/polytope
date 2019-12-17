#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/Constants.h"

namespace Tests {

   namespace Vector {
      using Polytope::Epsilon;
      using Polytope::Vector;

      namespace Equality {
         TEST(Equality, Equals) {
            Vector p1 = Vector(1.0f, 1.0f, 1.0f);
            Vector p2 = Vector(1.0f, 1.0f, 1.0f);

            EXPECT_EQ(p1, p1);
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p2, p2);
         }

         TEST(Equality, NotEquals) {
            Vector p1 = Vector(1.0f, 1.0f, 1.0f);
            Vector p2 = Vector(0.0f, 1.0f, 1.0f);
            EXPECT_NE(p1, p2);
         }
      }

      namespace Dot {
         TEST(Dot, Dot1) {
            Vector p1 = Vector(1, 2, 3);
            Vector p2 = Vector(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 32.0f;

            EXPECT_NEAR(actual, expected, Epsilon);
         }

         TEST(Dot, Dot2) {
            Vector p1 = Vector(-1, 2, 3);
            Vector p2 = Vector(4, 5, 6);

            float actual = p1.Dot(p2);
            float expected = 24.0f;

            EXPECT_NEAR(actual, expected, Epsilon);
         }

         TEST(Dot, Dot3) {
            Vector p1 = Vector(0, 0, 0);
            Vector p2 = Vector(0, 0, 0);

            float actual = p1.Dot(p2);
            float expected = 0.f;

            EXPECT_NEAR(actual, expected, Epsilon);
         }
      }
   }
}