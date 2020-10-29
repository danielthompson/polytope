////
//// Created by daniel on 10/27/20.
////
//#include "gtest/gtest.h"
//#include "../src/common/utilities/thread_pool.h"
//
//namespace Tests {
//   TEST(thread_pool_test, enqueue_one_sleep) {
//
//      int i = 0;
//
//      {
//         poly::thread_pool pool(1);
//         pool.enqueue([&i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      ASSERT_EQ(i, 1);
//   }
//
//   TEST(thread_pool_test, enqueue_two_sleep) {
//
//      int i = 0;
//      int j = 0;
//
//      {
//         poly::thread_pool pool(2);
//         pool.enqueue([&i] {
//            usleep(100000);
//            i = 1;
//         });
//         pool.enqueue([&j] {
//            usleep(100000);
//            j = 1;
//         });
//      }
//
//      EXPECT_EQ(i, 1);
//      EXPECT_EQ(j, 1);
//   }
//
//   TEST(thread_pool_test, enqueue_many_sleep) {
//
//      int array[128];
//      memset(array, 0, sizeof(int) * 128);
//
//      {
//         poly::thread_pool pool(32);
//         for (int & i : array) {
//            pool.enqueue([&array, &i] {
//               usleep(100000);
//               i = 1;
//            });
//         }
//      }
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//   }
//
//   TEST(thread_pool_test, child_enqueue) {
//      poly::thread_pool pool(4);
//
//      std::function<void(int)> task = [&task, &pool](int depth) {
//         usleep(100000);
//         printf("task depth %i\n", depth);
//
//         if (depth == 0)
//            return;
//
//         std::function<void()> child_wrapper = [&task, depth] {
//            task(depth - 1);
//         };
//
//         pool.enqueue(child_wrapper);
//      };
//
//      std::function<void()> wrapper = [&task] { task(10); };
//
//      pool.enqueue(wrapper);
//   }
//
//   TEST(thread_pool_test, child_enqueue_with_vector) {
//      poly::thread_pool pool(4);
//
//      std::function<void(int, std::vector<int>&)> task = [&task, &pool](int depth, std::vector<int> &vector) {
//         usleep(100000);
//         //printf("task depth %i\n", depth);
//
//         if (depth == 0)
//            return;
//
//         vector.push_back(depth);
//
//         std::function<void()> child_wrapper = [&task, &vector, depth] {
//            task(depth - 1, vector);
//         };
//
//         pool.enqueue(child_wrapper);
//      };
//
//      std::vector<int> v;
//
//      std::function<void()> wrapper = [&] { task(10, v); };
//
//      pool.enqueue(wrapper);
//      pool.synchronize();
//
//      ASSERT_EQ(v.size(), 10);
//      ASSERT_EQ(v[0], 10);
//      ASSERT_EQ(v[1], 9);
//      ASSERT_EQ(v[2], 8);
//      ASSERT_EQ(v[3], 7);
//      ASSERT_EQ(v[4], 6);
//      ASSERT_EQ(v[5], 5);
//      ASSERT_EQ(v[6], 4);
//      ASSERT_EQ(v[7], 3);
//      ASSERT_EQ(v[8], 2);
//      ASSERT_EQ(v[9], 1);
//   }
//
//   struct test_node {
//      test_node* left_child;
//      test_node* right_child;
//   };
//
//   TEST(thread_pool_test, child_enqueue_with_tree) {
//      poly::thread_pool pool(4);
//
//      std::function<void(int, struct test_node*)> task = [&task, &pool](int depth, struct test_node* node) {
//         usleep(100000);
//         //printf("task depth %i\n", depth);
//
//         if (depth == 0)
//            return;
//
//         struct test_node* left_child = new test_node();
//         struct test_node* right_child = new test_node();
//
//         node->left_child = left_child;
//         node->right_child = right_child;
//
//         std::function<void()> child_wrapper = [&task, left_child, depth] {
//            task(depth - 1, left_child);
//         };
//
//         pool.enqueue(child_wrapper);
//
//         child_wrapper = [&task, right_child, depth] {
//            task(depth - 1, right_child);
//         };
//
//         pool.enqueue(child_wrapper);
//      };
//
//      struct test_node* root_node = new test_node();
//
//      std::function<void()> wrapper = [&] { task(2, root_node); };
//
//      pool.enqueue(wrapper);
//      pool.synchronize();
//
//      ASSERT_NE(root_node->left_child, nullptr);
//      ASSERT_NE(root_node->right_child, nullptr);
//
//      ASSERT_NE(root_node->left_child->left_child, nullptr);
//      ASSERT_NE(root_node->left_child->right_child, nullptr);
//      ASSERT_NE(root_node->right_child->left_child, nullptr);
//      ASSERT_NE(root_node->right_child->right_child, nullptr);
//
//      ASSERT_EQ(root_node->left_child->left_child->left_child, nullptr);
//      ASSERT_EQ(root_node->left_child->left_child->right_child, nullptr);
//      ASSERT_EQ(root_node->left_child->right_child->left_child, nullptr);
//      ASSERT_EQ(root_node->left_child->right_child->right_child, nullptr);
//      ASSERT_EQ(root_node->right_child->left_child->left_child, nullptr);
//      ASSERT_EQ(root_node->right_child->left_child->right_child, nullptr);
//      ASSERT_EQ(root_node->right_child->right_child->left_child, nullptr);
//      ASSERT_EQ(root_node->right_child->right_child->right_child, nullptr);
//   }
//
//   TEST(thread_pool_test, run_to_completion_one_thread_one_element) {
//
//      int array[1];
//      memset(array, 0, sizeof(int) * 1);
//
//      poly::thread_pool pool(1);
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//   }
//
//   TEST(thread_pool_test, run_to_completion_one_thread_one_element_twice) {
//
//      int array[1];
//      memset(array, 0, sizeof(int) * 1);
//
//      poly::thread_pool pool(1);
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 0;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 0);
//      }
//   }
//
//   TEST(thread_pool_test, run_to_completion_one_thread_many_elements) {
//
//      int array[16];
//      memset(array, 0, sizeof(int) * 16);
//
//      poly::thread_pool pool(1);
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//   }
//
//   TEST(thread_pool_test, run_to_completion_one_thread_many_elements_twice) {
//
//      int array[16];
//      memset(array, 0, sizeof(int) * 16);
//
//      poly::thread_pool pool(1);
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 0;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 0);
//      }
//   }
//
//
//   TEST(thread_pool_test, run_to_completion_many_threads_one_element) {
//
//      int array[1];
//      memset(array, 0, sizeof(int) * 1);
//
//      poly::thread_pool pool(2);
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//   }
//
//   TEST(thread_pool_test, run_to_completion_many_threads_one_element_twice) {
//
//      int array[1];
//      memset(array, 0, sizeof(int) * 1);
//
//      poly::thread_pool pool(2);
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 0;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 0);
//      }
//   }
//
//   TEST(thread_pool_test, run_to_completion_many_threads_many_elements) {
//
//      int array[16];
//      memset(array, 0, sizeof(int) * 16);
//
//      poly::thread_pool pool(4);
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//   }
//
//
//   TEST(thread_pool_test, run_to_completion_many_threads_many_elements_twice) {
//
//      int array[16];
//      memset(array, 0, sizeof(int) * 16);
//
//      poly::thread_pool pool(4);
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 1;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 1);
//      }
//
//      for (int & i : array) {
//         pool.enqueue([&array, &i] {
//            usleep(100000);
//            i = 0;
//         });
//      }
//
//      pool.synchronize();
//
//      for (int & i : array) {
//         EXPECT_EQ(i, 0);
//      }
//   }
//}
