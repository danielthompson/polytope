#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/shapes/AbstractShape.h"
#include "../src/structures/Intersection.h"

namespace Tests {

   namespace BoundingBox {
      using Polytope::Point;
      using Polytope::Vector;
      using Polytope::Ray;
      using Polytope::Intersection;
      using Polytope::BoundingBox;

      namespace Intersect {
         TEST(BoundingBox, Intersect) {
            Point min(0.0f, 0.0f, 0.0f);
            Point max(1.0f, 1.0f, 1.0f);

            BoundingBox b(min, max);

            Point origin(0.5, 0.5, 10);
            Vector direction(0, 0, -1);

            Ray ray(origin, direction);

            Intersection intersection;
            b.Intersect(ray, &intersection);

            EXPECT_TRUE(intersection.Hits);
            EXPECT_EQ(ray.MinT, 9);
         }
      }
   }
}
