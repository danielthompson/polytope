//
// Created by daniel on 10/31/20.
//

#include "gtest/gtest.h"

#include "../../../src/common/parsers/PBRTFileParser.h"
#include "../../../src/cpu/films/PNGFilm.h"
#include "../../../src/cpu/filters/BoxFilter.h"
#include "../../../src/cpu/integrators/PathTraceIntegrator.h"
#include "../../../src/cpu/cameras/PerspectiveCamera.h"

namespace Tests {
   TEST(Instancing, Teapot) {
   
      auto fp = poly::PBRTFileParser();
      std::string file = "../scenes/test/instancing/teapot-instancing.pbrt";
      std::unique_ptr<poly::AbstractRunner> runner = fp.ParseFile(file);
   
      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      ASSERT_EQ(1, runner->Scene->Shapes.size());
      
      poly::Mesh* mesh = runner->Scene->Shapes[0];
      ASSERT_NE(nullptr, mesh);
      
      auto geometry = mesh->mesh_geometry;
      ASSERT_NE(nullptr, geometry);
      
      ASSERT_FALSE(geometry->has_vertex_normals);
      EXPECT_EQ(2256, geometry->num_faces);
      EXPECT_EQ(1177, geometry->num_vertices_packed);
      
   }
}