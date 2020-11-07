//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../src/common/utilities/Logger.h"
#include "../../src/common/parsers/mesh_parsers.h"

namespace Tests {

   namespace Parse {
      TEST(PLYParser, Teapot) {

         const poly::ply_parser parser;
         const std::string file = "../scenes/teapot/teapot.ply";
         auto geometry = std::make_shared<poly::mesh_geometry>();

         parser.parse_file(geometry, file);

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
         auto ply_geometry = std::make_shared<poly::mesh_geometry>();
         {
            const poly::ply_parser ply_parser;
            const std::string file = "../scenes/teapot/teapot_converted.ply";
            ply_parser.parse_file(ply_geometry, file);
         }

         auto obj_geometry = std::make_shared<poly::mesh_geometry>();
         {
            const poly::obj_parser obj_parser;
            const std::string file = "../scenes/teapot/teapot.obj";
            obj_parser.parse_file(obj_geometry, file);
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
         const poly::ply_parser parser;
         constexpr unsigned int expected_num_vertices = 8;
         constexpr unsigned int expected_num_faces = 4;

         // binary file
         const std::string binary_file = "../scenes/test/floor-binary-le.ply";
         auto binary_geometry = std::make_shared<poly::mesh_geometry>();

         parser.parse_file(binary_geometry, binary_file);
         // ensure nothing is null
         ASSERT_NE(nullptr, binary_geometry);
         EXPECT_EQ(expected_num_vertices, binary_geometry->num_vertices_packed);
         EXPECT_EQ(expected_num_faces, binary_geometry->num_faces);
         
         // ascii file
         const std::string ascii_file = "../scenes/test/floor-ascii.ply";
         auto ascii_geometry = std::make_shared<poly::mesh_geometry>();

         parser.parse_file(ascii_geometry, ascii_file);
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

      void simple_parse_helper(const std::string& file) {
         const poly::ply_parser parser;
         auto geometry = std::make_shared<poly::mesh_geometry>();
         parser.parse_file(geometry, file);

         // ensure nothing is null
         ASSERT_NE(nullptr, geometry);

         ASSERT_EQ(3, geometry->num_vertices_packed);
         const poly::Point v0 = geometry->get_vertex(0);

         EXPECT_EQ(1, v0.x);
         EXPECT_EQ(2, v0.y);
         EXPECT_EQ(3, v0.z);

         const poly::Point v1 = geometry->get_vertex(1);

         EXPECT_EQ(4, v1.x);
         EXPECT_EQ(5, v1.y);
         EXPECT_EQ(6, v1.z);

         const poly::Point v2 = geometry->get_vertex(2);

         EXPECT_EQ(7, v2.x);
         EXPECT_EQ(8, v2.y);
         EXPECT_EQ(9, v2.z);

         // faces
         ASSERT_EQ(1, geometry->num_faces);
      }
      
      TEST(PLYParser, parse_xyz) {
         simple_parse_helper("../scenes/test/ply_parsing/xyz.ply");
      }

      TEST(PLYParser, parse_xzy) {
         simple_parse_helper("../scenes/test/ply_parsing/xzy.ply");
      }
      
      TEST(PLYParser, element_vertex_first) {
         simple_parse_helper("../scenes/test/ply_parsing/element-vertex-first.ply");
      }

      TEST(PLYParser, element_face_first) {
         simple_parse_helper("../scenes/test/ply_parsing/element-face-first.ply");
      }

      TEST(PLYParser, header_1) {
         using poly::ply_parser;
         
         const poly::ply_parser parser;
         auto geometry = std::make_shared<poly::mesh_geometry>();
         std::string file = "../scenes/test/ply_parsing/element-vertex-first.ply";
         poly::ply_parser::parser_state state = parser.parse_header(file);

         EXPECT_EQ(ply_parser::ply_format::ascii, state.data_format);
         EXPECT_FALSE( state.has_vertex_normals);
         EXPECT_FALSE(state.elements.empty());
         ASSERT_EQ(2, state.elements.size());
         
         auto& first_element = state.elements[0];
         EXPECT_EQ(ply_parser::ply_element_type::vertex, first_element.type);
         EXPECT_FALSE(first_element.properties.empty());
         EXPECT_EQ(3, first_element.properties.size());
         EXPECT_EQ(3, first_element.num_instances);
         
         auto& vertex_property_x = first_element.properties[0];
         EXPECT_EQ(ply_parser::ply_property_type::ply_float, vertex_property_x.type);
         EXPECT_EQ(ply_parser::ply_property_name::x, vertex_property_x.name);

         auto& vertex_property_y = first_element.properties[1];
         EXPECT_EQ(ply_parser::ply_property_type::ply_float, vertex_property_y.type);
         EXPECT_EQ(ply_parser::ply_property_name::y, vertex_property_y.name);

         auto& vertex_property_z = first_element.properties[2];
         EXPECT_EQ(ply_parser::ply_property_type::ply_float, vertex_property_z.type);
         EXPECT_EQ(ply_parser::ply_property_name::z, vertex_property_z.name);
         
         auto& second_element = state.elements[1];
         EXPECT_EQ(ply_parser::ply_element_type::face, second_element.type);
         EXPECT_FALSE(second_element.properties.empty());
         EXPECT_EQ(1, second_element.properties.size());
         
         auto& face_property = second_element.properties[0];
         EXPECT_EQ(ply_parser::ply_property_type::ply_uchar, face_property.list_prefix_type);
         EXPECT_EQ(ply_parser::ply_property_type::ply_int, face_property.list_elements_type);
      }

      TEST(PLYParser, header_2) {
         using poly::ply_parser;

         const poly::ply_parser parser;
         auto geometry = std::make_shared<poly::mesh_geometry>();
         std::string file = "../scenes/test/ply_parsing/element-face-first.ply";
         poly::ply_parser::parser_state state = parser.parse_header(file);

         EXPECT_EQ(ply_parser::ply_format::ascii, state.data_format);
         EXPECT_FALSE( state.has_vertex_normals);
         EXPECT_FALSE(state.elements.empty());
         ASSERT_EQ(2, state.elements.size());

         auto& first_element = state.elements[0];
         EXPECT_EQ(ply_parser::ply_element_type::face, first_element.type);
         EXPECT_FALSE(first_element.properties.empty());
         EXPECT_EQ(1, first_element.properties.size());

         auto& face_property = first_element.properties[0];
         EXPECT_EQ(ply_parser::ply_property_type::ply_uchar, face_property.list_prefix_type);
         EXPECT_EQ(ply_parser::ply_property_type::ply_int, face_property.list_elements_type);
         
         auto& second_element = state.elements[1];
         EXPECT_EQ(ply_parser::ply_element_type::vertex, second_element.type);
         EXPECT_FALSE(second_element.properties.empty());
         EXPECT_EQ(3, second_element.properties.size());
         EXPECT_EQ(3, second_element.num_instances);

         auto& vertex_property_x = second_element.properties[0];
         EXPECT_EQ(ply_parser::ply_property_type::ply_float, vertex_property_x.type);
         EXPECT_EQ(ply_parser::ply_property_name::x, vertex_property_x.name);

         auto& vertex_property_y = second_element.properties[1];
         EXPECT_EQ(ply_parser::ply_property_type::ply_float, vertex_property_y.type);
         EXPECT_EQ(ply_parser::ply_property_name::y, vertex_property_y.name);

         auto& vertex_property_z = second_element.properties[2];
         EXPECT_EQ(ply_parser::ply_property_type::ply_float, vertex_property_z.type);
         EXPECT_EQ(ply_parser::ply_property_name::z, vertex_property_z.name);
      }

      TEST(PLYParser, header_geometry_agreement1) {

         const poly::ply_parser parser;
         const std::string file = "../scenes/test/ply_parsing/dragon_vrip-normals-cleaned.ply";
         auto geometry = std::make_shared<poly::mesh_geometry>();

         parser.parse_file(geometry, file);

         // ensure nothing is null
         ASSERT_NE(nullptr, geometry);

         ASSERT_EQ(437645, geometry->num_vertices_packed);
         ASSERT_EQ(871306, geometry->num_faces);
         ASSERT_EQ(2613918, geometry->num_vertices);
      }

      TEST(PLYParser, header_geometry_agreement2) {

         const poly::ply_parser parser;
         const std::string file = "../scenes/test/ply_parsing/dragon_vrip-normals-cleaned-binary.ply";
         auto geometry = std::make_shared<poly::mesh_geometry>();

         parser.parse_file(geometry, file);

         // ensure nothing is null
         ASSERT_NE(nullptr, geometry);

         ASSERT_EQ(437645, geometry->num_vertices_packed);
         ASSERT_EQ(871306, geometry->num_faces);
         ASSERT_EQ(2613918, geometry->num_vertices);
      }
   }
}
