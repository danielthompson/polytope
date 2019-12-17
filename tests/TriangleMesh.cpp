#include <iostream>
#include "gtest/gtest.h"

#include "../src/structures/Point.h"
#include "../src/Constants.h"

namespace Tests {

   using Polytope::Epsilon;
   using Polytope::Point;

   namespace Equality {

// IndependentMethod is a tests case - here, we have 2 tests for this 1 tests case
      TEST(Equality, Equals) {
         Point p1 = Point(1.0f, 1.0f, 1.0f);
         Point p2 = Point(1.0f, 1.0f, 1.0f);
         EXPECT_EQ(p1, p2);
      }

      TEST(Equality, NotEquals) {
         Point p1 = Point(1.0f, 1.0f, 1.0f);
         Point p2 = Point(0.0f, 1.0f, 1.0f);
         EXPECT_NE(p1, p2);
      }

// The fixture for testing class Project1. From google tests primer.
      class Project1Test : public ::testing::Test {
      protected:
         // You can remove any or all of the following functions if its body
         // is empty.

         Project1Test() {
            // You can do set-up work for each tests here.
         }

         virtual ~Project1Test() {
            // You can do clean-up work that doesn't throw exceptions here.
         }

         // If the constructor and destructor are not enough for setting up
         // and cleaning up each tests, you can define the following methods:
         virtual void SetUp() {
            // Code here will be called immediately after the constructor (right
            // before each tests).
         }

         virtual void TearDown() {
            // Code here will be called immediately after each tests (right
            // before the destructor).
         }

         // Objects declared here can be used by all tests in the tests case for Project1.
         Point p = Point(0, 0, 0);
      };

// Test case must be called the class above
// Also note: use TEST_F instead of TEST to access the tests fixture (from google tests primer)
      TEST_F(Project1Test, MethodBarDoesAbc) {
         Point p2 = Point(0.0f, 0.0f, 0.0f);
         EXPECT_EQ(p, p2);
      }
   }

   namespace Dot {
      TEST(Dot, Dot1) {
         Point p1 = Point(1, 2, 3);
         Point p2 = Point(4, 5, 6);

         float actual = p1.Dot(p2);
         float expected = 32.0f;

         EXPECT_NEAR(actual, expected, Epsilon);
      }

      TEST(Dot, Dot2) {
         Point p1 = Point(-1, 2, 3);
         Point p2 = Point(4, 5, 6);

         float actual = p1.Dot(p2);
         float expected = 24.0f;

         EXPECT_NEAR(actual, expected, Epsilon);
      }

      TEST(Dot, Dot3) {
         Point p1 = Point(0, 0, 0);
         Point p2 = Point(0, 0, 0);

         float actual = p1.Dot(p2);
         float expected = 0.f;

         EXPECT_NEAR(actual, expected, Epsilon);
      }
   }
}