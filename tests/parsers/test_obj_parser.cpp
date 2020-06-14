//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../src/common/utilities/Logger.h"
#include "../../src/common/parsers/mesh/OBJParser.h"
#include "../../src/cpu/shapes/mesh.h"


namespace Tests {

   namespace Parse {
      
      void test_parse_obj_helper(poly::Mesh* mesh) {
         const poly::OBJParser parser;
         const std::string file = "../scenes/teapot/teapot.obj";
         parser.ParseFile(mesh, file);

         // ensure nothing is null
         ASSERT_NE(nullptr, mesh);
         ASSERT_EQ(3644, mesh->num_vertices_packed);

         // check a random-ish vertex for correctness
         const poly::Point secondToLastVertex = mesh->get_vertex(3642);

         // EXPECT_FLOAT_EQ allows 4 ulps difference
         EXPECT_FLOAT_EQ(3.428125, secondToLastVertex.x);
         EXPECT_FLOAT_EQ(2.477344, secondToLastVertex.y);
         EXPECT_FLOAT_EQ(0.000000, secondToLastVertex.z);

         // faces
         ASSERT_EQ(6320, mesh->num_faces);

         // check a random-ish face for correctness
         const poly::Point3ui secondToLastFace = mesh->get_vertex_indices_for_face(6318);

         EXPECT_EQ(3021, secondToLastFace.x);
         EXPECT_EQ(3020, secondToLastFace.y);
         EXPECT_EQ(3000, secondToLastFace.z);
      }
      
      TEST(OBJParser, Mesh) {
         const std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
         poly::Mesh mesh(identity, identity, nullptr);
         test_parse_obj_helper(&mesh);
      }
   }
}
