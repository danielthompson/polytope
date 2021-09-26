#include "gtest/gtest.h"

#include "../src/cpu/constants.h"
#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/transform.h"
#include "../src/cpu/structures/stats.h"

struct poly::stats main_stats;
thread_local struct poly::stats thread_stats;

namespace Tests {


   
   namespace Common {

      TEST(Common, SignedDistanceFromPlane1) {
         poly::point pointOnPlane(0, 0, 0);
         poly::normal normal(1, 1, 1);
         normal.normalize();

         poly::point p(1, 1, 1);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane2) {
         poly::point pointOnPlane(0, 0, 0);
         poly::normal normal(1, 1, 1);
         normal.normalize();

         poly::point p(-1, -1, -1);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, -poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane3) {
         poly::point pointOnPlane(0, 0, 0);
         poly::normal normal(1, 1, 1);
         normal.normalize();

         poly::point p(1, 1, 4);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, 2 * poly::Root3);
      }

      TEST(Common, SignedDistanceFromPlane4) {
         poly::point pointOnPlane(0, 0, 0);
         poly::normal normal(0, 1, 0);
         normal.normalize();

         main_stats.num_bb_intersections_miss++;
         thread_stats.num_bb_intersections_miss++;
         
         poly::point p(1, 15, 4);

         float actual = poly::SignedDistanceFromPlane(pointOnPlane, normal, p);
         EXPECT_EQ(actual, 15);
      }
   }
}
