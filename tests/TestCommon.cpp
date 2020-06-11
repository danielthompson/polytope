#include "gtest/gtest.h"

#include "../src/cpu/constants.h"
#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/Transform.h"
#include "../src/cpu/shapes/abstract_mesh.h"

namespace Tests {

   namespace Common {
      using poly::Normal;
      using poly::Point;
      using poly::Vector;
      using poly::Transform;
      using poly::AbstractMesh;

      TEST(Common, SignedDistanceFromPlane1) {
         Point pointOnPlane(0, 0, 0);
         Normal normal(1, 1, 1);
         normal.Normalize();

         Point p(1, 1, 1);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane2) {
         Point pointOnPlane(0, 0, 0);
         Normal normal(1, 1, 1);
         normal.Normalize();

         Point p(-1, -1, -1);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, -poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane3) {
         Point pointOnPlane(0, 0, 0);
         Normal normal(1, 1, 1);
         normal.Normalize();

         Point p(1, 1, 4);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, 2 * poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane4) {
         Point pointOnPlane(0, 0, 0);
         Normal normal(0, 1, 0);
         normal.Normalize();

         Point p(1, 15, 4);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, 15);
      }
   }
}
