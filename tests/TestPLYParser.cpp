//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../src/utilities/Logger.h"
#include "../src/parsers/mesh/PLYParser.h"
#include "../src/structures/Vectors.h"

namespace Tests {

   namespace Parse {
      TEST(PLYParser, Teapot) {

         const Polytope::PLYParser parser;
         const std::string file = "../scenes/teapot/teapot.ply";
         const std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::TriangleMesh* mesh = new Polytope::TriangleMesh(identity, identity, nullptr);

         parser.ParseFile(mesh, file);

         // ensure nothing is null
         ASSERT_NE(nullptr, mesh);

         ASSERT_EQ(1177, mesh->Vertices.size());

         // check a random-ish vertex for correctness
         const Polytope::Point secondToLastVertex = mesh->Vertices[1175];

         // EXPECT_FLOAT_EQ allows 4 ulps difference

         EXPECT_FLOAT_EQ(0.313617, secondToLastVertex.x);
         EXPECT_FLOAT_EQ(0.087529, secondToLastVertex.y);
         EXPECT_FLOAT_EQ(2.98125, secondToLastVertex.z);

         // faces
         ASSERT_EQ(2256, mesh->Faces.size());

         // check a random-ish face for correctness
         const Polytope::Point3ui secondToLastFace = mesh->Faces[2254];

         EXPECT_EQ(623, secondToLastFace.x);
         EXPECT_EQ(1176, secondToLastFace.y);
         EXPECT_EQ(1087, secondToLastFace.z);

         delete mesh;
      }
   }
}
