#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/intersection.h"

namespace Tests {

   namespace Intersection {
      namespace Transformation {
         TEST(Intersection, WorldToLocal1) {
            poly::intersection intersection;
            intersection.bent_normal = poly::normal(0, 0, 1);
            intersection.tangent_1 = poly::vector(1, 0, 0);
            intersection.tangent_2 = poly::vector(0, 1, 0);

            poly::vector worldIncoming(0, 0, -1);

            poly::vector localIncoming = intersection.world_to_local(worldIncoming);

            EXPECT_EQ(localIncoming.x, 0);
            EXPECT_EQ(localIncoming.y, -1);
            EXPECT_EQ(localIncoming.z, 0);
         }

         TEST(Intersection, WorldToLocal2) {
            poly::intersection intersection;
            intersection.bent_normal = poly::normal(0, 0, 1);
            intersection.tangent_1 = poly::vector(0, 1, 0);
            intersection.tangent_2 = poly::vector(1, 0, 0);

            poly::vector worldIncoming(0, 0, -1);

            poly::vector localIncoming = intersection.world_to_local(worldIncoming);

            EXPECT_EQ(localIncoming.x, 0);
            EXPECT_EQ(localIncoming.y, -1);
            EXPECT_EQ(localIncoming.z, 0);
         }
      }
   }
}
