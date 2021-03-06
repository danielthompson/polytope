//
// Created by Daniel Thompson on 12/28/19.
//

#include <fstream>
#include "gtest/gtest.h"
#include "../../src/cpu/shapes/tesselators.h"
// #include "../src/exporters/OBJExporter.h"

namespace Tests {
   // TODO fix this up
//   class TesselatorTests : public ::testing::Test {
//   public:
//      void Output(const std::string& filename, poly::AbstractMesh* mesh) const {
//         std::ofstream filestream;
//         filestream.open (filename);
//         poly::OBJExporter exporter;
//         exporter.Export(filestream, mesh, false);
//         filestream.close();
//      }
//   };
//
//   class DiskTesselatorTests : public TesselatorTests {
//   protected:
//      void CreateTest(poly::AbstractMesh* mesh, const unsigned int meridians) {
//         poly::DiskTesselator tesselator;
//         tesselator.Create(meridians, mesh);
//
//         const unsigned int expectedVertices = meridians + 1;
//         const unsigned int expectedFaces = meridians;
//         EXPECT_EQ(mesh->Vertices.size(), expectedVertices);
//         EXPECT_EQ(mesh->Faces.size(), expectedFaces);
//
//         std::stringstream stringstream;
//         stringstream << "tests/disk" << meridians << ".obj";
//         Output(stringstream.str(), mesh);
//      }
//   };
//
//   TEST_F(DiskTesselatorTests, Create1) {
//      std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
//      for (unsigned int meridians = 3; meridians <= 30; meridians += 3) {
//         poly::TriangleMesh mesh = poly::TriangleMesh(identity, identity, nullptr);
//         CreateTest(&mesh, meridians);
//
//      }
//   }
//
//   class ConeTesselatorTests : public TesselatorTests {
//   protected:
//      void CreateTest(poly::AbstractMesh* mesh, const unsigned int meridians) {
//         poly::ConeTesselator tesselator;
//         tesselator.Create(meridians, mesh);
//
//         const unsigned int expectedVertices = meridians + 1;
//         const unsigned int expectedFaces = meridians;
//         EXPECT_EQ(mesh->Vertices.size(), expectedVertices);
//         EXPECT_EQ(mesh->Faces.size(), expectedFaces);
//
//         std::stringstream stringstream;
//         stringstream << "tests/cone" << meridians << ".obj";
//         Output(stringstream.str(), mesh);
//      }
//   };
//
//   TEST_F(ConeTesselatorTests, Create1) {
//      std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
//      for (unsigned int meridians = 3; meridians <= 30; meridians += 3) {
//         poly::TriangleMesh mesh = poly::TriangleMesh(identity, identity, nullptr);
//         CreateTest(&mesh, meridians);
//      }
//   }
//
//
//   class SphereTesselatorTests : public TesselatorTests {
//   protected:
//      void CreateTest(poly::AbstractMesh* mesh, const unsigned int meridians, const unsigned int parallels) {
//         poly::SphereTesselator tesselator;
//         tesselator.Create(meridians, parallels, mesh);
//
//         const unsigned int expectedVertices = 2 + meridians * parallels;
//         const unsigned int expectedFaces = 2 * meridians * parallels;
//         EXPECT_EQ(mesh->Vertices.size(), expectedVertices);
//         EXPECT_EQ(mesh->Faces.size(), expectedFaces);
//
//         std::stringstream stringstream;
//         stringstream << "tests/sphere" << meridians << "x" << parallels << ".obj";
//         Output(stringstream.str(), mesh);
//      }
//   };
//
//   TEST_F(SphereTesselatorTests, Create1) {
//      std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
//      for (unsigned int meridians = 3; meridians <= 30; meridians += 3) {
//         for (unsigned int parallels = 3; parallels <= 30; parallels += 3) {
//            poly::TriangleMesh mesh = poly::TriangleMesh(identity, identity, nullptr);
//            CreateTest(&mesh, meridians, parallels);
//         }
//      }
//   }
}
