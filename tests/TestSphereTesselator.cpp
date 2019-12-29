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
      void CreateTest(Polytope::TriangleMesh* mesh, const unsigned int meridians, const unsigned int parallels) {
         Polytope::SphereTesselator tesselator;
         tesselator.Create(meridians, parallels, mesh);

         Polytope::OBJExporter exporter;

         const unsigned int expectedVertices = 2 + meridians * parallels;
         const unsigned int expectedFaces = 2 * meridians * parallels;
         EXPECT_EQ(mesh->Vertices.size(), expectedVertices);
         EXPECT_EQ(mesh->Faces.size(), expectedFaces);

         std::stringstream stringstream;
         stringstream << "tests/sphere" << meridians << "x" << parallels << ".obj";
         std::ofstream filestream;
         filestream.open (stringstream.str());
         exporter.Export(filestream, mesh, false);
         filestream.close();
      }
   };

   TEST_F(SphereTesselatorTests, Create1) {
      std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();


//      const int meridians = 3;
//      const int parallels = 1;

      for (unsigned int meridians = 3; meridians <= 30; meridians += 3) {
         for (unsigned int parallels = 3; parallels <= 30; parallels += 3) {
            Polytope::TriangleMesh mesh = Polytope::TriangleMesh(identity, identity, nullptr);
            CreateTest(&mesh, meridians, parallels);
         }
      }

//      CreateTest(&mesh, 4, 4);
//      CreateTest(&mesh, 20, 1);


   }
}
