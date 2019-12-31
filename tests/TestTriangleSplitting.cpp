#include <fstream>
#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/shapes/TriangleMesh.h"
#include "../src/exporters/OBJExporter.h"

namespace Tests {

   namespace TriangleSplitting {
      using Polytope::Normal;
      using Polytope::Point;
      using Polytope::Vector;
      using Polytope::Transform;
      using Polytope::TriangleMesh;

      TEST(TriangleSplitting, CutY1) {
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

      TEST(TriangleSplitting, CutY2) {
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

      
   }
}
