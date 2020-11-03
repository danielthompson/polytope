#include "gtest/gtest.h"
#include "../src/cpu/acceleration/bvh.h"

namespace Tests {
   namespace bvh {
      TEST(bvh, compact1) {
         poly::bvh bvh;
         bvh.root = new poly::bvh_node();
         bvh.root->bb.p1.x = 0;
         
         bvh.num_nodes = 1;
         
         bvh.compact();

         poly::compact_bvh* compact = bvh.compact_root;
         ASSERT_NE(compact, nullptr);
         ASSERT_EQ(compact->num_nodes, 1);
         
         EXPECT_EQ(compact->nodes[0].bb.p1.x, 0);
      }

      TEST(bvh, compact3) {
         poly::bvh bvh;
         bvh.root = new poly::bvh_node();
         bvh.root->bb.p1.x = 0;

         bvh.root->low = new poly::bvh_node();
         bvh.root->low->bb.p1.x = 1;

         bvh.root->high = new poly::bvh_node();
         bvh.root->high->bb.p1.x = 2;

         bvh.num_nodes = 3;

         bvh.compact();

         poly::compact_bvh* compact = bvh.compact_root;
         ASSERT_NE(compact, nullptr);
         ASSERT_EQ(compact->num_nodes, 3);

         EXPECT_EQ(compact->nodes[0].bb.p1.x, 0);
         EXPECT_EQ(compact->nodes[0].high_child_offset, 2);

         EXPECT_EQ(compact->nodes[1].bb.p1.x, 1);

         EXPECT_EQ(compact->nodes[2].bb.p1.x, 2);
      }

      TEST(bvh, compact5left) {
         poly::bvh bvh;
         bvh.root = new poly::bvh_node();
         bvh.root->bb.p1.x = 0;

         bvh.root->low = new poly::bvh_node();
         bvh.root->low->bb.p1.x = 1;

         bvh.root->low->low = new poly::bvh_node();
         bvh.root->low->low->bb.p1.x = 2;

         bvh.root->low->high = new poly::bvh_node();
         bvh.root->low->high->bb.p1.x = 3;

         bvh.root->high = new poly::bvh_node();
         bvh.root->high->bb.p1.x = 4;

         bvh.num_nodes = 5;

         bvh.compact();

         poly::compact_bvh* compact = bvh.compact_root;
         ASSERT_NE(compact, nullptr);
         ASSERT_EQ(compact->num_nodes, 5);

         EXPECT_EQ(compact->nodes[0].bb.p1.x, 0);
         EXPECT_EQ(compact->nodes[0].high_child_offset, 4);

         EXPECT_EQ(compact->nodes[1].bb.p1.x, 1);
         EXPECT_EQ(compact->nodes[1].high_child_offset, 2);

         EXPECT_EQ(compact->nodes[2].bb.p1.x, 2);
         EXPECT_EQ(compact->nodes[3].bb.p1.x, 3);
         EXPECT_EQ(compact->nodes[4].bb.p1.x, 4);
      }

      TEST(bvh, compact5right) {
         poly::bvh bvh;
         bvh.root = new poly::bvh_node();
         bvh.root->bb.p1.x = 0;

         bvh.root->low = new poly::bvh_node();
         bvh.root->low->bb.p1.x = 1;

         bvh.root->high = new poly::bvh_node();
         bvh.root->high->bb.p1.x = 2;

         bvh.root->high->low = new poly::bvh_node();
         bvh.root->high->low->bb.p1.x = 3;

         bvh.root->high->high = new poly::bvh_node();
         bvh.root->high->high->bb.p1.x = 4;
         
         bvh.num_nodes = 5;

         bvh.compact();

         poly::compact_bvh* compact = bvh.compact_root;
         ASSERT_NE(compact, nullptr);
         ASSERT_EQ(compact->num_nodes, 5);

         EXPECT_EQ(compact->nodes[0].bb.p1.x, 0);
         EXPECT_EQ(compact->nodes[0].high_child_offset, 2);

         EXPECT_EQ(compact->nodes[1].bb.p1.x, 1);
         EXPECT_EQ(compact->nodes[2].bb.p1.x, 2);
         EXPECT_EQ(compact->nodes[2].high_child_offset, 2);
         
         EXPECT_EQ(compact->nodes[3].bb.p1.x, 3);
         EXPECT_EQ(compact->nodes[4].bb.p1.x, 4);
      }
      
      TEST(bvh, compact7full) {
         poly::bvh bvh;
         bvh.root = new poly::bvh_node();
         bvh.root->bb.p1.x = 0;

         bvh.root->low = new poly::bvh_node();
         bvh.root->low->bb.p1.x = 1;

         bvh.root->low->low = new poly::bvh_node();
         bvh.root->low->low->bb.p1.x = 2;

         bvh.root->low->high = new poly::bvh_node();
         bvh.root->low->high->bb.p1.x = 3;

         bvh.root->high = new poly::bvh_node();
         bvh.root->high->bb.p1.x = 4;

         bvh.root->high->low = new poly::bvh_node();
         bvh.root->high->low->bb.p1.x = 5;

         bvh.root->high->high = new poly::bvh_node();
         bvh.root->high->high->bb.p1.x = 6;

         bvh.num_nodes = 7;

         bvh.compact();

         poly::compact_bvh* compact = bvh.compact_root;
         ASSERT_NE(compact, nullptr);
         ASSERT_EQ(compact->num_nodes, 7);

         EXPECT_EQ(compact->nodes[0].bb.p1.x, 0);
         EXPECT_EQ(compact->nodes[0].high_child_offset, 4);

         EXPECT_EQ(compact->nodes[1].bb.p1.x, 1);
         EXPECT_EQ(compact->nodes[1].high_child_offset, 2);

         EXPECT_EQ(compact->nodes[2].bb.p1.x, 2);
         EXPECT_EQ(compact->nodes[3].bb.p1.x, 3);
         EXPECT_EQ(compact->nodes[4].bb.p1.x, 4);
         EXPECT_EQ(compact->nodes[5].bb.p1.x, 5);
         EXPECT_EQ(compact->nodes[6].bb.p1.x, 6);
      }
      
      TEST(bvh, bound1) {
         std::shared_ptr<poly::mesh_geometry> geometry = std::make_shared<poly::mesh_geometry>(); 
         
         geometry->add_vertex(0, 0, 0);
         geometry->add_vertex(1, 0, 0);
         geometry->add_vertex(0, 1, 0);
         geometry->add_packed_face(0, 1, 2);

         auto identity = std::make_shared<poly::Transform>();
         
         poly::Mesh mesh(identity, identity, nullptr, geometry);
         std::vector<poly::Mesh *> meshes = {
               &mesh
         };
         
         poly::bvh bvh;
         bvh.bound(meshes);
         
         ASSERT_NE(bvh.root, nullptr);
      }

      TEST(bvh, bound2) {
         std::shared_ptr<poly::mesh_geometry> geometry = std::make_shared<poly::mesh_geometry>();
         

         unsigned int num_faces = 4;
         float offset = 2.f;
         for (unsigned int mesh_index = 0; mesh_index < num_faces; mesh_index++) {
            geometry->add_vertex(0 + (mesh_index * offset), 0, 0);
            geometry->add_vertex(1 + (mesh_index * offset), 0, 0);
            geometry->add_vertex(0 + (mesh_index * offset), 1, 0);
            geometry->add_packed_face(3 * mesh_index, 3 * mesh_index + 1, 3 * mesh_index + 2);
         }

         auto identity = std::make_shared<poly::Transform>();
         poly::Mesh mesh(identity, identity, nullptr, geometry);
         
         std::vector<poly::Mesh *> meshes = {
               &mesh
         };

         poly::bvh bvh;
         bvh.bound(meshes);

         ASSERT_NE(bvh.root, nullptr);
         ASSERT_NE(bvh.root->low, nullptr);
         ASSERT_NE(bvh.root->high, nullptr);
      }
   }
}
