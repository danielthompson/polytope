//
// Created by Daniel Thompson on 12/28/19.
//

#include "gtest/gtest.h"
#include "../src/shapes/TriangleMesh.h"
#include "../src/shapes/SphereTesselator.h"

namespace Tests {
   namespace Equality {

      using Polytope::SphereTesselator;
      using Polytope::Transform;
      using Polytope::TriangleMesh;

      TEST(SphereTesselator, Create1) {

         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh = TriangleMesh(identity, identity, nullptr);

         SphereTesselator tesselator;
         tesselator.Create(3, 1, &mesh);

         EXPECT_EQ(mesh.Vertices.size(), 5);
         EXPECT_EQ(mesh.Faces.size(), 6);

         for (const auto &point : mesh.Vertices) {
            std::cout << point.x << " " << point.y << " " << point.z << std::endl;
         }

      }

      TEST(SphereTesselator, Create2) {

         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh = TriangleMesh(identity, identity, nullptr);

         SphereTesselator tesselator;
         tesselator.Create(3, 2, &mesh);

         EXPECT_EQ(mesh.Vertices.size(), 8);
         EXPECT_EQ(mesh.Faces.size(), 6);

         for (const auto &point : mesh.Vertices) {
            std::cout << point.x << " " << point.y << " " << point.z << std::endl;
         }
      }

      TEST(SphereTesselator, Create3) {

         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh = TriangleMesh(identity, identity, nullptr);

         SphereTesselator tesselator;
         tesselator.Create(12, 12, &mesh);

         EXPECT_EQ(mesh.Vertices.size(), 8);
         EXPECT_EQ(mesh.Faces.size(), 6);

         for (const auto &point : mesh.Vertices) {
            std::cout << point.x << " " << point.y << " " << point.z << std::endl;
         }
      }
   }
}

