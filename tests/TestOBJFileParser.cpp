//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../src/utilities/Logger.h"
#include "../src/parsers/mesh/OBJFileParser.h"
#include "../src/structures/Vectors.h"

namespace Tests {

   namespace Parse {
      TEST(OBJParser, Teapot) {

         const Polytope::OBJFileParser parser;
         const std::string file = "../scenes/teapot/teapot.obj";
         const std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::TriangleMesh* mesh = new Polytope::TriangleMesh(identity, identity, nullptr);

         parser.ParseFile(mesh, file);

         // ensure nothing is null
         ASSERT_NE(nullptr, mesh);

         ASSERT_EQ(3644, mesh->Vertices.size());

         // check a random-ish vertex for correctness
         const Polytope::Point secondToLastVertex = mesh->Vertices[3642];

         // EXPECT_FLOAT_EQ allows 4 ulps difference

         EXPECT_FLOAT_EQ(3.428125, secondToLastVertex.x);
         EXPECT_FLOAT_EQ(2.477344, secondToLastVertex.y);
         EXPECT_FLOAT_EQ(0.000000, secondToLastVertex.z);

         // faces
         ASSERT_EQ(6320, mesh->Faces.size());

         // check a random-ish face for correctness
         const Polytope::Point3ui secondToLastFace = mesh->Faces[6318];

         EXPECT_EQ(3021, secondToLastFace.x);
         EXPECT_EQ(3020, secondToLastFace.y);
         EXPECT_EQ(3000, secondToLastFace.z);
         delete mesh;
      }
   }
}
