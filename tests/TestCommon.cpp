#include "gtest/gtest.h"

#include "../src/Constants.h"
#include "../src/structures/Vectors.h"
#include "../src/shapes/TriangleMesh.h"


namespace Tests {

   namespace Common {
      using Polytope::Normal;
      using Polytope::Point;
      using Polytope::Vector;
      using Polytope::Transform;
      using Polytope::TriangleMesh;

      TEST(Common, SignedDistanceFromPlane1) {
         Point pointOnPlane(0, 0, 0);
         Normal normal(1, 1, 1);
         normal.Normalize();

         Point p(1, 1, 1);

         float actual = Polytope::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, Polytope::Root3);
      }

      TEST(Common, SignedDistanceFromPlane2) {
         Point pointOnPlane(0, 0, 0);
         Normal normal(1, 1, 1);
         normal.Normalize();

         Point p(-1, -1, -1);

         float actual = Polytope::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, -Polytope::Root3);
      }

      TEST(Common, SignedDistanceFromPlane3) {
         Point pointOnPlane(0, 0, 0);
         Normal normal(1, 1, 1);
         normal.Normalize();

         Point p(1, 1, 4);

         float actual = Polytope::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, 2 * Polytope::Root3);
      }

      TEST(Common, SignedDistanceFromPlane4) {
         Point pointOnPlane(0, 0, 0);
         Normal normal(0, 1, 0);
         normal.Normalize();

         Point p(1, 15, 4);

         float actual = Polytope::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, 15);
      }
   }
}
