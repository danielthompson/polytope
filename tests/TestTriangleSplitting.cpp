#include <fstream>
#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/shapes/TriangleMesh.h"
#include "../src/exporters/OBJExporter.h"
#include "../src/shapes/Tesselators.h"

namespace Tests {

   namespace TriangleSplitting {
      using Polytope::Normal;
      using Polytope::Point;
      using Polytope::Vector;
      using Polytope::Transform;
      using Polytope::TriangleMesh;

      TEST(TriangleSplitting, SplitY1) {
         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh(identity, identity, nullptr);
         mesh.Vertices.emplace_back(0, 0, 0);
         mesh.Vertices.emplace_back(1, 0, 0);
         mesh.Vertices.emplace_back(0, 1, 0);
         mesh.Faces.emplace_back(0, 1, 2);

         mesh.SplitY(0);

         EXPECT_EQ(mesh.Vertices.size(), 3);
         EXPECT_EQ(mesh.Faces.size(), 1);
      }

      TEST(TriangleSplitting, SplitY2) {
         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh(identity, identity, nullptr);
         mesh.Vertices.emplace_back(0, 1, 0);
         mesh.Vertices.emplace_back(0, 0, 0);
         mesh.Vertices.emplace_back(1, 0, 0);
         mesh.Faces.emplace_back(0, 1, 2);

         std::ofstream filestream;
         Polytope::OBJExporter exporter;
         filestream.open ("before.obj");
         exporter.Export(filestream, &mesh, false);
         filestream.close();

         mesh.SplitY(0.5f);

         filestream.open ("after.obj");
         exporter.Export(filestream, &mesh, false);
         filestream.close();


         EXPECT_EQ(mesh.Vertices.size(), 5);
         EXPECT_EQ(mesh.Faces.size(), 3);
      }

      TEST(TriangleSplitting, SplitY3) {
         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh(identity, identity, nullptr);
         mesh.Vertices.emplace_back(0, 0, 0);
         mesh.Vertices.emplace_back(1, 0, 0);
         mesh.Vertices.emplace_back(0, 1, 0);
         mesh.Faces.emplace_back(0, 1, 2);

         std::ofstream filestream;
         Polytope::OBJExporter exporter;
         filestream.open ("before.obj");
         exporter.Export(filestream, &mesh, false);
         filestream.close();

         mesh.SplitY(0.5f);

         filestream.open ("after.obj");
         exporter.Export(filestream, &mesh, false);
         filestream.close();

         EXPECT_EQ(mesh.Vertices.size(), 5);
         EXPECT_EQ(mesh.Faces.size(), 3);
      }

      TEST(TriangleSplitting, SplitY4) {

         // edge case in which the splitting plane exactly intersects a vertex:
         //
         //  |\
         //  | \
         //  |  \
         // -|--->---
         //  |  /
         //  | /
         //  |/
         //
         // in this case, we should split the triangle into 2, not 3.

         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh(identity, identity, nullptr);
         mesh.Vertices.emplace_back(1, 0, 0);
         mesh.Vertices.emplace_back(0, -1, 0);
         mesh.Vertices.emplace_back(0, 1, 0);
         mesh.Faces.emplace_back(0, 1, 2);

         std::ofstream filestream;
         Polytope::OBJExporter exporter;
         filestream.open ("before.obj");
         exporter.Export(filestream, &mesh, false);
         filestream.close();

         mesh.SplitY(0.f);

         filestream.open ("after.obj");
         exporter.Export(filestream, &mesh, false);
         filestream.close();

         EXPECT_EQ(mesh.Vertices.size(), 4);
         EXPECT_EQ(mesh.Faces.size(), 2);
      }

      TEST(TriangleSplitting, SplitCone) {
         std::shared_ptr<Transform> identity = std::make_shared<Transform>();
         TriangleMesh mesh(identity, identity, nullptr);

         Polytope::SphereTesselator tesselator;
         tesselator.Create(15, 15, &mesh);

         std::ofstream filestream;
         Polytope::OBJExporter exporter;
         filestream.open ("before-sphere.obj");
         exporter.Export(filestream, &mesh, false);
         filestream.close();

         const Point pointOnPlane(0, 0, 0);
         Normal normal(1, 1, 0);
         normal.Normalize();

         mesh.Split(pointOnPlane, normal);

         filestream.open ("after-sphere.obj");
         exporter.Export(filestream, &mesh, false);
         filestream.close();

         EXPECT_EQ(mesh.Vertices.size(), 4);
         EXPECT_EQ(mesh.Faces.size(), 2);
      }
   }
}
