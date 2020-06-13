#include "gtest/gtest.h"

#include "../src/cpu/constants.h"
#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/Transform.h"

namespace Tests {

   namespace Common {

      TEST(Common, SignedDistanceFromPlane1) {
         poly::Point pointOnPlane(0, 0, 0);
         poly::Normal normal(1, 1, 1);
         normal.Normalize();

         poly::Point p(1, 1, 1);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane2) {
         poly::Point pointOnPlane(0, 0, 0);
         poly::Normal normal(1, 1, 1);
         normal.Normalize();

         poly::Point p(-1, -1, -1);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, -poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane3) {
         poly::Point pointOnPlane(0, 0, 0);
         poly::Normal normal(1, 1, 1);
         normal.Normalize();

         poly::Point p(1, 1, 4);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, 2 * poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane4) {
         poly::Point pointOnPlane(0, 0, 0);
         poly::Normal normal(0, 1, 0);
         normal.Normalize();

         poly::Point p(1, 15, 4);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, 15);
      }
   }
}
