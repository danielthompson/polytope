//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../src/utilities/Logger.h"
#include "../../src/parsers/mesh/PLYParser.h"
#include "../../src/structures/Vectors.h"
#include "../../src/parsers/mesh/OBJParser.h"

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

      TEST(PLYParser, TeapotConverted) {

         const std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::TriangleMeshSOA* ply_mesh = new Polytope::TriangleMeshSOA(identity, identity, nullptr);
         {
            const Polytope::PLYParser ply_parser;
            const std::string file = "../scenes/teapot/teapot_converted.ply";
            ply_parser.ParseFile(ply_mesh, file);
         }

         Polytope::TriangleMeshSOA* obj_mesh = new Polytope::TriangleMeshSOA(identity, identity, nullptr);
         {
            const Polytope::OBJParser obj_parser;
            const std::string file = "../scenes/teapot/teapot.obj";
            obj_parser.ParseFile(obj_mesh, file);
         }
            
         ASSERT_FALSE(ply_mesh == nullptr);
         ASSERT_FALSE(obj_mesh == nullptr);

         EXPECT_EQ(ply_mesh->x.size(), obj_mesh->x.size());
         
         constexpr float epsilon = 0.001;
         
         for (unsigned int i = 0; i < ply_mesh->x.size(); i++) {
            const float delta = std::abs(ply_mesh->x_expanded[i] - obj_mesh->x_expanded[i]);
            EXPECT_TRUE(delta < epsilon);
            //EXPECT_FLOAT_EQ(ply_mesh->x_expanded[i], obj_mesh->x_expanded[i]);
         }
         
         EXPECT_EQ(ply_mesh->x_expanded.size(), obj_mesh->x_expanded.size());
         EXPECT_EQ(ply_mesh->y.size(), obj_mesh->y.size());
         EXPECT_EQ(ply_mesh->y_expanded.size(), obj_mesh->y_expanded.size());
         EXPECT_EQ(ply_mesh->z.size(), obj_mesh->z.size());
         EXPECT_EQ(ply_mesh->z_expanded.size(), obj_mesh->z_expanded.size());

         EXPECT_EQ(ply_mesh->nx.size(), obj_mesh->nx.size());
         EXPECT_EQ(ply_mesh->nx_expanded.size(), obj_mesh->nx_expanded.size());
         EXPECT_EQ(ply_mesh->ny.size(), obj_mesh->ny.size());
         EXPECT_EQ(ply_mesh->ny_expanded.size(), obj_mesh->ny_expanded.size());
         EXPECT_EQ(ply_mesh->nz.size(), obj_mesh->nz.size());
         EXPECT_EQ(ply_mesh->nz_expanded.size(), obj_mesh->nz_expanded.size());

         EXPECT_EQ(ply_mesh->num_vertices_expanded, obj_mesh->num_vertices_expanded);
         EXPECT_EQ(ply_mesh->num_vertices, obj_mesh->num_vertices);
         EXPECT_EQ(ply_mesh->num_faces, obj_mesh->num_faces);

         ASSERT_EQ(ply_mesh->fv0.size(), obj_mesh->fv0.size());

         for (unsigned int i = 0; i < ply_mesh->fv0.size(); i++) {
            EXPECT_EQ(ply_mesh->fv0[i], obj_mesh->fv0[i]);
         }
         
         ASSERT_EQ(ply_mesh->fv1.size(), obj_mesh->fv1.size());

         for (unsigned int i = 0; i < ply_mesh->fv0.size(); i++) {
            EXPECT_EQ(ply_mesh->fv1[i], obj_mesh->fv1[i]);
         }

         ASSERT_EQ(ply_mesh->fv2.size(), obj_mesh->fv2.size());

         for (unsigned int i = 0; i < ply_mesh->fv0.size(); i++) {
            EXPECT_EQ(ply_mesh->fv2[i], obj_mesh->fv2[i]);
         }
         
         delete ply_mesh;
         delete obj_mesh;
      }
      
      TEST(PLYParser, Binary) {
         const Polytope::PLYParser parser;
         const std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         constexpr unsigned int expected_num_vertices = 8;
         constexpr unsigned int expected_num_faces = 4;

         // binary file
         const std::string binary_file = "../scenes/test/floor-binary-le.ply";
         Polytope::TriangleMesh* binary_mesh = new Polytope::TriangleMesh(identity, identity, nullptr);

         parser.ParseFile(binary_mesh, binary_file);
         // ensure nothing is null
         ASSERT_NE(nullptr, binary_mesh);
         EXPECT_EQ(expected_num_vertices, binary_mesh->Vertices.size());
         EXPECT_EQ(expected_num_faces, binary_mesh->Faces.size());
         
         // ascii file
         const std::string ascii_file = "../scenes/test/floor-ascii.ply";
         Polytope::TriangleMesh* ascii_mesh = new Polytope::TriangleMesh(identity, identity, nullptr);

         parser.ParseFile(ascii_mesh, ascii_file);
         // ensure nothing is null
         ASSERT_NE(nullptr, ascii_mesh);
         EXPECT_EQ(expected_num_vertices, ascii_mesh->Vertices.size());
         EXPECT_EQ(expected_num_faces, ascii_mesh->Faces.size());

         // check vertices
         for (unsigned int i = 0; i < expected_num_vertices; i++) {
            EXPECT_EQ(binary_mesh->Vertices[i], ascii_mesh->Vertices[i]);
         }
         
         // check faces
         for (unsigned int i = 0; i < expected_num_faces; i++) {
            EXPECT_EQ(binary_mesh->Vertices[i], ascii_mesh->Vertices[i]);
         }

         delete ascii_mesh;
         delete binary_mesh;
      }
   }
}
