//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../src/utilities/Logger.h"
#include "../src/parsers/OBJFileParser.h"


namespace Tests {

   namespace Parse {
      TEST(FileParser, Teapot) {

         Polytope::OBJFileParser fp = Polytope::OBJFileParser();
         std::string file = "../scenes/teapot.obj";
         std::unique_ptr<Polytope::TriangleMesh> mesh = fp.ParseFile(file);

         // ensure nothing is null
         ASSERT_NE(nullptr, mesh);

         ASSERT_EQ(3644, mesh->Vertices.size());

         // check a random-ish vertex for correctness
         const Polytope::Point secondToLastVertex = mesh->Vertices[3642];

         ASSERT_EQ(3.428125, secondToLastVertex.x);
         ASSERT_EQ(2.477344, secondToLastVertex.y);
         ASSERT_EQ(0.000000, secondToLastVertex.z);

         // faces
         ASSERT_EQ(6320, mesh->Faces.size());

         // check a random-ish face for correctness
         const Polytope::Point3ui secondToLastFace = mesh->Faces[6318];

         ASSERT_EQ(3022, secondToLastFace.x);
         ASSERT_EQ(3021, secondToLastFace.y);
         ASSERT_EQ(3001, secondToLastFace.z);

      }
   }
}
