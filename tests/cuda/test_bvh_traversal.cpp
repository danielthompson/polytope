//
// Created by daniel on 10/16/20.
//

#include <gtest/gtest.h>
#include "../../src/cpu/scenes/Scene.h"
#include "../../src/common/parsers/mesh/PLYParser.h"
#include "../../src/cuda/gpu_memory_manager.h"
#include "../../src/cuda/kernels/path_tracer.cuh"

namespace Tests {

   /**
    * A ray pointed at a bounding box should hit that bounding box.
    */
   TEST(cuda_bvh_traversal, test1) {

      const poly::PLYParser parser;
      const std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();

      const std::string file = "../scenes/test/cuda/cuda-bvh-traversal1.ply";
      poly::Mesh mesh(identity, identity, nullptr);
      parser.ParseFile(&mesh, file);
      
      poly::Scene scene(nullptr);
      scene.Shapes.emplace_back(&mesh);

      scene.bvh_root.root = new poly::bvh_node();
      scene.bvh_root.root->bb = { { -3, -1, -1 }, {3, 2, 1} };
      scene.bvh_root.num_nodes = 1;
      scene.bvh_root.compact();
      
      poly::GPUMemoryManager memory_manager = poly::GPUMemoryManager(1, 1);
      size_t bytes_copied = memory_manager.MallocScene(&scene);
      EXPECT_GT(bytes_copied, 0);
      
      poly::PathTracerKernel kernel(&memory_manager);
      
      poly::Ray ray = { {2, 0, 10}, { 0, 0, -1}};
      
      bool actual = kernel.unit_test_hit_ray_against_bounding_box(ray, memory_manager.device_bvh->aabb);
      EXPECT_TRUE(actual) << "should hit root";
   }

   /**
    * A ray pointed at one child should hit the root, hit the child, and miss the other child.
    */
   TEST(cuda_bvh_traversal, test2) {

      const poly::PLYParser parser;
      const std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();

      const std::string file = "../scenes/test/cuda/cuda-bvh-traversal1.ply";
      poly::Mesh mesh(identity, identity, nullptr);
      parser.ParseFile(&mesh, file);

      poly::Scene scene(nullptr);
      scene.Shapes.emplace_back(&mesh);

      scene.bvh_root.root = new poly::bvh_node();
      scene.bvh_root.root->bb = { { -3, -1, -1 }, {3, 2, 1} };
      scene.bvh_root.root->axis = poly::Axis::x;
      
      // low child
      scene.bvh_root.root->low = new poly::bvh_node();
      scene.bvh_root.root->low->bb = { { -3, -1, -1 }, {-1, 2, 1} };
      
      // high child
      scene.bvh_root.root->high = new poly::bvh_node();
      scene.bvh_root.root->high->bb = { { 1, -1, -1 }, {3, 2, 1} };
      
      scene.bvh_root.num_nodes = 3;
      scene.bvh_root.compact();

      poly::GPUMemoryManager memory_manager = poly::GPUMemoryManager(1, 1);
      size_t bytes_copied = memory_manager.MallocScene(&scene);

      EXPECT_GT(bytes_copied, 0);
      
      poly::PathTracerKernel kernel(&memory_manager);

      poly::Ray ray = { {2, 0, 10}, { 0, 0, -1}};

      bool actual;
      
      actual = kernel.unit_test_hit_ray_against_bounding_box(ray, (memory_manager.device_bvh + 1)->aabb);
      EXPECT_FALSE(actual) << "shouldn't hit low child";
      
      actual = kernel.unit_test_hit_ray_against_bounding_box(ray, (memory_manager.device_bvh + 2)->aabb);
      EXPECT_TRUE(actual) << "should hit high child";
   }


   /**
    * A ray inside one child, pointed at the other child, should hit all three bounding boxes.
    */
   TEST(cuda_bvh_traversal, test3) {

      const poly::PLYParser parser;
      const std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();

      const std::string file = "../scenes/test/cuda/cuda-bvh-traversal1.ply";
      poly::Mesh mesh(identity, identity, nullptr);
      parser.ParseFile(&mesh, file);

      poly::Scene scene(nullptr);
      scene.Shapes.emplace_back(&mesh);

      scene.bvh_root.root = new poly::bvh_node();
      scene.bvh_root.root->bb = { { -3, -1, -1 }, {3, 2, 1} };
      scene.bvh_root.root->axis = poly::Axis::x;

      // low child
      scene.bvh_root.root->low = new poly::bvh_node();
      scene.bvh_root.root->low->bb = { { -3, -1, -1 }, {-1, 2, 1} };

      // high child
      scene.bvh_root.root->high = new poly::bvh_node();
      scene.bvh_root.root->high->bb = { { 1, -1, -1 }, {3, 2, 1} };

      scene.bvh_root.num_nodes = 3;
      scene.bvh_root.compact();

      poly::GPUMemoryManager memory_manager = poly::GPUMemoryManager(1, 1);
      size_t bytes_copied = memory_manager.MallocScene(&scene);

      EXPECT_GT(bytes_copied, 0);

      poly::PathTracerKernel kernel(&memory_manager);

      poly::Ray ray = { {2, 0, 0}, { -1, 0, 0}};

      bool actual;

      actual = kernel.unit_test_hit_ray_against_bounding_box(ray, memory_manager.device_bvh->aabb);
      EXPECT_TRUE(actual) << "should hit root";
      
      actual = kernel.unit_test_hit_ray_against_bounding_box(ray, (memory_manager.device_bvh + 1)->aabb);
      EXPECT_TRUE(actual) << "should hit low child";

      actual = kernel.unit_test_hit_ray_against_bounding_box(ray, (memory_manager.device_bvh + 2)->aabb);
      EXPECT_TRUE(actual) << "should hit high child";
   }
}