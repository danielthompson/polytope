//
// Created by Daniel Thompson on 12/28/19.
//

#include <fstream>
#include "gtest/gtest.h"
#include "../src/shapes/TriangleMesh.h"
#include "../src/shapes/SphereTesselator.h"
#include "../src/exporters/OBJExporter.h"

namespace Tests {
   class SphereTesselatorTests : public ::testing::Test {
   protected:
      void CreateTest(Polytope::TriangleMesh* mesh, const int meridians, const int parallels) {
         Polytope::SphereTesselator tesselator;
         tesselator.Create(meridians, parallels, mesh);

         EXPECT_EQ(mesh->Vertices.size(), 8);
         EXPECT_EQ(mesh->Faces.size(), 6);

         Polytope::OBJExporter exporter;

         std::stringstream stringstream;
         stringstream << "sphere" << meridians << "x" << parallels << ".obj";
         std::ofstream filestream;
         filestream.open (stringstream.str());
         exporter.Export(filestream, mesh, false);
         filestream.close();
      }
   };

   TEST_F(SphereTesselatorTests, Create1) {
      std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
      Polytope::TriangleMesh mesh = Polytope::TriangleMesh(identity, identity, nullptr);
      for (unsigned int meridians = 3; meridians < 15; meridians += 3) {
         for (unsigned int parallels = 1; parallels < 15; parallels += 3) {
            CreateTest(&mesh, meridians, parallels);
         }
      }
   }
}
