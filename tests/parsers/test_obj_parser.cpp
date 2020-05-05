//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../src/utilities/Logger.h"
#include "../../src/parsers/mesh/OBJParser.h"
#include "../../src/shapes/linear_aos/mesh_linear_aos.h"
#include "../../src/shapes/bvh_split/mesh_bvh_split.h"

namespace Tests {

   namespace Parse {
      
      void test_parse_obj_helper(Polytope::AbstractMesh* mesh) {
         const Polytope::OBJParser parser;
         const std::string file = "../scenes/teapot/teapot.obj";
         parser.ParseFile(mesh, file);

         // ensure nothing is null
         ASSERT_NE(nullptr, mesh);
         ASSERT_EQ(3644, mesh->num_vertices_packed);

         // check a random-ish vertex for correctness
         const Polytope::Point secondToLastVertex = mesh->get_vertex(3642);

         // EXPECT_FLOAT_EQ allows 4 ulps difference
         EXPECT_FLOAT_EQ(3.428125, secondToLastVertex.x);
         EXPECT_FLOAT_EQ(2.477344, secondToLastVertex.y);
         EXPECT_FLOAT_EQ(0.000000, secondToLastVertex.z);

         // faces
         ASSERT_EQ(6320, mesh->num_faces);

         // check a random-ish face for correctness
         const Polytope::Point3ui secondToLastFace = mesh->get_vertex_indices_for_face(6318);

         EXPECT_EQ(3021, secondToLastFace.x);
         EXPECT_EQ(3020, secondToLastFace.y);
         EXPECT_EQ(3000, secondToLastFace.z);
      }
      
      TEST(OBJParser, MeshLinearSOA) {
         const std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::MeshLinearAOS mesh(identity, identity, nullptr);
         test_parse_obj_helper(&mesh);
      }

      TEST(OBJParser, MeshLinearAOS) {
         const std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::MeshLinearAOS mesh(identity, identity, nullptr);
         test_parse_obj_helper(&mesh);
     }

      TEST(OBJParser, MeshBVHSplit) {
         const std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::MeshBVHSplit mesh(identity, identity, nullptr);
         test_parse_obj_helper(&mesh);
      }
   }
}
