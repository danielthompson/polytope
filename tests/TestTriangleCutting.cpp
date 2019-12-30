#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/shapes/TriangleMesh.h"

namespace Tests {

   namespace TriangleCutting {
      using Polytope::Normal;
      using Polytope::Point;
      using Polytope::Vector;
      using Polytope::Transform;
      using Polytope::TriangleMesh;

      TEST(TriangleCutting, CutY1) {
         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh(identity, identity, nullptr);
         mesh.Vertices.emplace_back(0, 0, 0);
         mesh.Vertices.emplace_back(1, 0, 0);
         mesh.Vertices.emplace_back(0, 1, 0);
         mesh.Faces.emplace_back(0, 1, 2);

         mesh.CutY(0);

         EXPECT_EQ(mesh.Vertices.size(), 3);
         EXPECT_EQ(mesh.Faces.size(), 1);
      }

      TEST(TriangleCutting, CutY2) {
         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh(identity, identity, nullptr);
         mesh.Vertices.emplace_back(0, 0, 0);
         mesh.Vertices.emplace_back(1, 0, 0);
         mesh.Vertices.emplace_back(0, 1, 0);
         mesh.Faces.emplace_back(0, 1, 2);

         mesh.CutY(0.5f);

         EXPECT_EQ(mesh.Vertices.size(), 5);
         EXPECT_EQ(mesh.Faces.size(), 3);
      }

      TEST(TriangleCutting, SignedDistance) {
         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh(identity, identity, nullptr);
         mesh.Vertices.emplace_back(0, 0, 0);
         mesh.Vertices.emplace_back(1, 0, 0);
         mesh.Vertices.emplace_back(0, 1, 0);
         mesh.Faces.emplace_back(0, 1, 2);

         mesh.CutY(0.5f);

         EXPECT_EQ(mesh.Vertices.size(), 5);
         EXPECT_EQ(mesh.Faces.size(), 3);
      }
   }
}
