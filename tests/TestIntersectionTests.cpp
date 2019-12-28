#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/shapes/AbstractShape.h"

namespace Tests {

   namespace Intersection {
      using Polytope::Intersection;
      using Polytope::Vector;
      using Polytope::Normal;
      using Polytope::Ray;
      using Polytope::BoundingBox;

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