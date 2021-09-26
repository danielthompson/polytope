//
// Created by daniel on 10/31/20.
//

#include "gtest/gtest.h"

#include "../../../src/common/parsers/pbrt_parser.h"
#include "../../../src/cpu/films/PNGFilm.h"
#include "../../../src/cpu/filters/BoxFilter.h"
#include "../../../src/cpu/integrators/PathTraceIntegrator.h"
#include "../../../src/cpu/cameras/perspective_camera.h"

namespace Tests {
   TEST(instancing_scene, teapot1) {
   
      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing1.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);
   
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

   TEST(instancing_scene, teapot2) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing2.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      ASSERT_EQ(2, runner->Scene->Shapes.size());

      poly::Mesh* mesh0 = runner->Scene->Shapes[0];
      ASSERT_NE(nullptr, mesh0);

      auto geometry0 = mesh0->mesh_geometry;
      ASSERT_NE(nullptr, geometry0);

      ASSERT_FALSE(geometry0->has_vertex_normals);
      EXPECT_EQ(2256, geometry0->num_faces);
      EXPECT_EQ(1177, geometry0->num_vertices_packed);

      poly::Mesh* mesh1 = runner->Scene->Shapes[1];
      ASSERT_NE(nullptr, mesh1);

      auto geometry1 = mesh1->mesh_geometry;
      ASSERT_NE(nullptr, geometry1);

      ASSERT_FALSE(geometry1->has_vertex_normals);
      EXPECT_EQ(2256, geometry1->num_faces);
      EXPECT_EQ(1177, geometry1->num_vertices_packed);

      ASSERT_EQ(geometry0, geometry1);
   }

   TEST(instancing_scene, teapot3) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing3.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      ASSERT_EQ(2, runner->Scene->Shapes.size());

      poly::Mesh* mesh0 = runner->Scene->Shapes[0];
      ASSERT_NE(nullptr, mesh0);

      auto geometry0 = mesh0->mesh_geometry;
      ASSERT_NE(nullptr, geometry0);

      ASSERT_FALSE(geometry0->has_vertex_normals);
      EXPECT_EQ(2256, geometry0->num_faces);
      EXPECT_EQ(1177, geometry0->num_vertices_packed);

      poly::Mesh* mesh1 = runner->Scene->Shapes[1];
      ASSERT_NE(nullptr, mesh1);

      auto geometry1 = mesh1->mesh_geometry;
      ASSERT_NE(nullptr, geometry1);

      ASSERT_FALSE(geometry1->has_vertex_normals);
      EXPECT_EQ(2256, geometry1->num_faces);
      EXPECT_EQ(1177, geometry1->num_vertices_packed);

      ASSERT_NE(geometry0, geometry1);
   }

   TEST(instancing_scene, teapot4a) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing4.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      ASSERT_EQ(2, runner->Scene->Shapes.size());

      poly::Mesh* mesh0 = runner->Scene->Shapes[0];
      ASSERT_NE(nullptr, mesh0);

      auto geometry0 = mesh0->mesh_geometry;
      ASSERT_NE(nullptr, geometry0);

      ASSERT_FALSE(geometry0->has_vertex_normals);
      EXPECT_EQ(2256, geometry0->num_faces);
      EXPECT_EQ(1177, geometry0->num_vertices_packed);

      poly::Mesh* mesh1 = runner->Scene->Shapes[1];
      ASSERT_NE(nullptr, mesh1);

      auto geometry1 = mesh1->mesh_geometry;
      ASSERT_NE(nullptr, geometry1);

      ASSERT_FALSE(geometry1->has_vertex_normals);
      EXPECT_EQ(2256, geometry1->num_faces);
      EXPECT_EQ(1177, geometry1->num_vertices_packed);

      ASSERT_NE(geometry0, geometry1);
   }

   /**
    * TODO Ensure that if any geometries are specified by an ObjectBegin / ObjectEnd directive block
    * and are not instanced by any ObjectInstance directive, they are removed from the scene after
    * parsing.
    */
   TEST(instancing_scene, teapot4b) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing4.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      ASSERT_EQ(2, runner->Scene->Shapes.size());

      poly::Mesh* mesh0 = runner->Scene->Shapes[0];
      ASSERT_NE(nullptr, mesh0);

      auto geometry0 = mesh0->mesh_geometry;
      ASSERT_NE(nullptr, geometry0);

      ASSERT_FALSE(geometry0->has_vertex_normals);
      EXPECT_EQ(2256, geometry0->num_faces);
      EXPECT_EQ(1177, geometry0->num_vertices_packed);

      poly::Mesh* mesh1 = runner->Scene->Shapes[1];
      ASSERT_NE(nullptr, mesh1);

      auto geometry1 = mesh1->mesh_geometry;
      ASSERT_NE(nullptr, geometry1);

      ASSERT_FALSE(geometry1->has_vertex_normals);
      EXPECT_EQ(2256, geometry1->num_faces);
      EXPECT_EQ(1177, geometry1->num_vertices_packed);

      ASSERT_NE(geometry0, geometry1);
   }

   /**
    * Test number of shapes in scene
    */
   TEST(instancing_scene, teapot4c) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing4.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      EXPECT_EQ(2, runner->Scene->Shapes.size());
   }

   /**
    * Test number of geometries in scene
    * TODO - remove unused geometries that are never instanced
    */
   TEST(instancing_scene, teapot4d) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing4.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      EXPECT_EQ(3, runner->Scene->mesh_geometry_count);
   }
   
   /**
    * Test number of shapes in scene
    */
   TEST(instancing_scene, teapot5a) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing5.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      EXPECT_EQ(3, runner->Scene->Shapes.size());
   }

   /**
    * Test number of geometries in scene
    */
   TEST(instancing_scene, teapot5b) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/instancing/teapot-instancing5.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      EXPECT_EQ(2, runner->Scene->mesh_geometry_count);
   }
}