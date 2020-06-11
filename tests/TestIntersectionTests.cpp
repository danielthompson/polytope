#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/shapes/abstract_mesh.h"

namespace Tests {

   namespace Intersection {
      using poly::Intersection;
      using poly::Vector;
      using poly::Normal;
      using poly::Ray;
      using poly::BoundingBox;

      namespace Transformation {
         TEST(Intersection, WorldToLocal1) {
            Intersection intersection;
            intersection.Normal = Normal(0, 0, 1);
            intersection.Tangent1 = Vector(1, 0, 0);
            intersection.Tangent2 = Vector(0, 1, 0);

            Vector worldIncoming(0, 0, -1);

            Vector localIncoming = intersection.WorldToLocal(worldIncoming);

            EXPECT_EQ(localIncoming.x, 0);
            EXPECT_EQ(localIncoming.y, -1);
            EXPECT_EQ(localIncoming.z, 0);
         }

         TEST(Intersection, WorldToLocal2) {
            Intersection intersection;
            intersection.Normal = Normal(0, 0, 1);
            intersection.Tangent1 = Vector(0, 1, 0);
            intersection.Tangent2 = Vector(1, 0, 0);

            Vector worldIncoming(0, 0, -1);

            Vector localIncoming = intersection.WorldToLocal(worldIncoming);

            EXPECT_EQ(localIncoming.x, 0);
            EXPECT_EQ(localIncoming.y, -1);
            EXPECT_EQ(localIncoming.z, 0);
         }
      }
   }
}
