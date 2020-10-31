//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../src/common/utilities/Logger.h"
#include "../../src/common/parsers/mesh/PLYParser.h"
#include "../../src/cpu/structures/Vectors.h"
#include "../../src/common/parsers/mesh/OBJParser.h"
#include "../../src/cpu/shapes/mesh.h"

namespace Tests {

   namespace Parse {
      TEST(PLYParser, Teapot) {

         const poly::PLYParser parser;
         const std::string file = "../scenes/teapot/teapot.ply";
         const std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
         auto geometry = std::make_shared<poly::mesh_geometry>();
         
         parser.ParseFile(geometry, file);

         // ensure nothing is null
         ASSERT_NE(nullptr, geometry);

         ASSERT_EQ(1177, geometry->num_vertices_packed);

         // check a random-ish vertex for correctness
         const poly::Point secondToLastVertex = geometry->get_vertex(1175);

         // EXPECT_FLOAT_EQ allows 4 ulps difference

         EXPECT_FLOAT_EQ(0.313617, secondToLastVertex.x);
         EXPECT_FLOAT_EQ(0.087529, secondToLastVertex.y);
         EXPECT_FLOAT_EQ(2.98125, secondToLastVertex.z);

         // faces
         ASSERT_EQ(2256, geometry->num_faces);

         // check a random-ish face for correctness
         const poly::Point3ui secondToLastFace = geometry->get_vertex_indices_for_face(2254);

         EXPECT_EQ(623, secondToLastFace.x);
         EXPECT_EQ(1176, secondToLastFace.y);
         EXPECT_EQ(1087, secondToLastFace.z);
      }

      TEST(PLYParser, TeapotConverted) {

         const std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
         auto ply_geometry = std::make_shared<poly::mesh_geometry>();
         {
            const poly::PLYParser ply_parser;
            const std::string file = "../scenes/teapot/teapot_converted.ply";
            ply_parser.ParseFile(ply_geometry, file);
         }

         auto obj_geometry = std::make_shared<poly::mesh_geometry>();
         {
            const poly::OBJParser obj_parser;
            const std::string file = "../scenes/teapot/teapot.obj";
            obj_parser.ParseFile(obj_geometry, file);
         }
            
         ASSERT_FALSE(ply_geometry == nullptr);
         ASSERT_FALSE(obj_geometry == nullptr);

         EXPECT_EQ(ply_geometry->x_packed.size(), obj_geometry->x_packed.size());
         
         constexpr float epsilon = 0.001;
         
         for (unsigned int i = 0; i < ply_geometry->x_packed.size(); i++) {
            const float delta = std::abs(ply_geometry->x[i] - obj_geometry->x[i]);
            EXPECT_TRUE(delta < epsilon);
            //EXPECT_FLOAT_EQ(ply_geometry->x_expanded[i], obj_geometry->x_expanded[i]);
         }
         
         EXPECT_EQ(ply_geometry->x.size(), obj_geometry->x.size());
         EXPECT_EQ(ply_geometry->y_packed.size(), obj_geometry->y_packed.size());
         EXPECT_EQ(ply_geometry->y.size(), obj_geometry->y.size());
         EXPECT_EQ(ply_geometry->z_packed.size(), obj_geometry->z_packed.size());
         EXPECT_EQ(ply_geometry->z.size(), obj_geometry->z.size());

         EXPECT_EQ(ply_geometry->nx_packed.size(), obj_geometry->nx_packed.size());
         EXPECT_EQ(ply_geometry->nx.size(), obj_geometry->nx.size());
         EXPECT_EQ(ply_geometry->ny_packed.size(), obj_geometry->ny_packed.size());
         EXPECT_EQ(ply_geometry->ny.size(), obj_geometry->ny.size());
         EXPECT_EQ(ply_geometry->nz_packed.size(), obj_geometry->nz_packed.size());
         EXPECT_EQ(ply_geometry->nz.size(), obj_geometry->nz.size());

         EXPECT_EQ(ply_geometry->num_vertices, obj_geometry->num_vertices);
         EXPECT_EQ(ply_geometry->num_vertices_packed, obj_geometry->num_vertices_packed);
         EXPECT_EQ(ply_geometry->num_faces, obj_geometry->num_faces);

         ASSERT_EQ(ply_geometry->fv0.size(), obj_geometry->fv0.size());

         for (unsigned int i = 0; i < ply_geometry->fv0.size(); i++) {
            EXPECT_EQ(ply_geometry->fv0[i], obj_geometry->fv0[i]);
         }
         
         ASSERT_EQ(ply_geometry->fv1.size(), obj_geometry->fv1.size());

         for (unsigned int i = 0; i < ply_geometry->fv0.size(); i++) {
            EXPECT_EQ(ply_geometry->fv1[i], obj_geometry->fv1[i]);
         }

         ASSERT_EQ(ply_geometry->fv2.size(), obj_geometry->fv2.size());

         for (unsigned int i = 0; i < ply_geometry->fv0.size(); i++) {
            EXPECT_EQ(ply_geometry->fv2[i], obj_geometry->fv2[i]);
         }
      }
      
      TEST(PLYParser, Binary) {
         const poly::PLYParser parser;
         const std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
         constexpr unsigned int expected_num_vertices = 8;
         constexpr unsigned int expected_num_faces = 4;

         // binary file
         const std::string binary_file = "../scenes/test/floor-binary-le.ply";
         auto binary_geometry = std::make_shared<poly::mesh_geometry>();

         parser.ParseFile(binary_geometry, binary_file);
         // ensure nothing is null
         ASSERT_NE(nullptr, binary_geometry);
         EXPECT_EQ(expected_num_vertices, binary_geometry->num_vertices_packed);
         EXPECT_EQ(expected_num_faces, binary_geometry->num_faces);
         
         // ascii file
         const std::string ascii_file = "../scenes/test/floor-ascii.ply";
         auto ascii_geometry = std::make_shared<poly::mesh_geometry>();

         parser.ParseFile(ascii_geometry, ascii_file);
         // ensure nothing is null
         ASSERT_NE(nullptr, ascii_geometry);
         EXPECT_EQ(expected_num_vertices, ascii_geometry->num_vertices_packed);
         EXPECT_EQ(expected_num_faces, ascii_geometry->num_faces);

         // check vertices
         for (unsigned int i = 0; i < expected_num_vertices; i++) {
            EXPECT_EQ(binary_geometry->get_vertex(i), ascii_geometry->get_vertex(i));
         }

         // TODO         
//         // check faces
//         for (unsigned int i = 0; i < expected_num_faces; i++) {
//            EXPECT_EQ(binary_mesh->Vertices[i], ascii_mesh->Vertices[i]);
//         }
      }
   }
}
