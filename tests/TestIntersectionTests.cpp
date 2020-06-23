#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/Intersection.h"

namespace Tests {

   namespace Intersection {
      namespace Transformation {
         TEST(Intersection, WorldToLocal1) {
            poly::Intersection intersection;
            intersection.geo_normal = poly::Normal(0, 0, 1);
            intersection.Tangent1 = poly::Vector(1, 0, 0);
            intersection.Tangent2 = poly::Vector(0, 1, 0);

            poly::Vector worldIncoming(0, 0, -1);

            poly::Vector localIncoming = intersection.WorldToLocal(worldIncoming);

            EXPECT_EQ(localIncoming.x, 0);
            EXPECT_EQ(localIncoming.y, -1);
            EXPECT_EQ(localIncoming.z, 0);
         }

         TEST(Intersection, WorldToLocal2) {
            poly::Intersection intersection;
            intersection.geo_normal = poly::Normal(0, 0, 1);
            intersection.Tangent1 = poly::Vector(0, 1, 0);
            intersection.Tangent2 = poly::Vector(1, 0, 0);

            poly::Vector worldIncoming(0, 0, -1);

            poly::Vector localIncoming = intersection.WorldToLocal(worldIncoming);

            EXPECT_EQ(localIncoming.x, 0);
            EXPECT_EQ(localIncoming.y, -1);
            EXPECT_EQ(localIncoming.z, 0);
         }
      }
   }
}
