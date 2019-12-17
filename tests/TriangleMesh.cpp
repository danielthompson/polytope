#include <iostream>
#include "gtest/gtest.h"

#include "../src/structures/Point.h"
#include "../src/Constants.h"
#include "../src/shapes/TriangleMesh.h"

namespace Tests {

   namespace TriangleMesh {

      TEST(TriangleMesh, Intersection) {
         Polytope::Transform identity;
         Polytope::TriangleMesh tm(identity, nullptr);

         tm.Vertices.emplace_back(Polytope::Point(0, 0, 0));
         tm.Vertices.emplace_back(Polytope::Point(0, 1, 0));
         tm.Vertices.emplace_back(Polytope::Point(1, 0, 0));

         tm.Faces.emplace_back(Polytope::Point3ui(0, 1, 2));

         Polytope::Ray hitRay(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, -1));
         bool actualHit = tm.Hits(hitRay);

         Polytope::Ray missRay(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, -1));
         bool actualMiss = tm.Hits(missRay);

         EXPECT_TRUE(actualHit);
         EXPECT_FALSE(actualMiss);
      }
   }
}
