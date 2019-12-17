#include "gtest/gtest.h"

#include "../src/structures/Point.h"
#include "../src/Constants.h"
#include "../src/shapes/TriangleMesh.h"

namespace Tests {

   namespace TriangleMesh {

      TEST(TriangleMesh, Hit) {
         Polytope::Transform identity;
         Polytope::TriangleMesh tm(identity, nullptr);

         tm.Vertices.emplace_back(Polytope::Point(0, 0, 0));
         tm.Vertices.emplace_back(Polytope::Point(0, 1, 0));
         tm.Vertices.emplace_back(Polytope::Point(1, 0, 0));

         tm.Faces.emplace_back(Polytope::Point3ui(0, 1, 2));

         // hits, from either direction

         Polytope::Ray ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, 1));
         EXPECT_TRUE(tm.Hits(ray));

         ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, -1));
         EXPECT_TRUE(tm.Hits(ray));

         ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, 1));
         EXPECT_FALSE(tm.Hits(ray));

         ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, -1));
         EXPECT_FALSE(tm.Hits(ray));
      }
   }
}
