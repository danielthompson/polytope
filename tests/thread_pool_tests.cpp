//
// Created by daniel on 10/27/20.
//
#include "gtest/gtest.h"
#include "../src/common/utilities/thread_pool.h"

namespace Tests {
   TEST(thread_pool_test, enqueue_one_sleep) {

      int i = 0;
      
      {
         poly::thread_pool pool(1);
         pool.enqueue([&i] {
            usleep(100000);
            i = 1;
         });
      }
      
      ASSERT_EQ(i, 1);
   }

   TEST(thread_pool_test, enqueue_two_sleep) {

      int i = 0;
      int j = 0;

      {
         poly::thread_pool pool(2);
         pool.enqueue([&i] {
            usleep(100000);
            i = 1;
         });
         pool.enqueue([&j] {
            usleep(100000);
            j = 1;
         });
      }

      EXPECT_EQ(i, 1);
      EXPECT_EQ(j, 1);
   }

   TEST(thread_pool_test, enqueue_many_sleep) {

      int array[128];
      memset(array, 0, sizeof(int) * 128);

      {
         poly::thread_pool pool(32);
         for (int & i : array) {
            pool.enqueue([&array, &i] {
               usleep(100000);
               i = 1;
            });
         }
      }

      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }
   }

   TEST(thread_pool_test, child_enqueue) {
      poly::thread_pool pool(32);
      
      std::function<void(int)> task = [&task, &pool](int depth) {
         usleep(100000);
         printf("task depth %i\n", depth);
         
         if (depth == 0)
            return;
         
         std::function<void()> child_wrapper = [&task, depth] {
            task(depth - 1);
         };
         
         pool.enqueue(child_wrapper);
      };
      
      std::function<void()> wrapper = [&task] { task(10); };

      pool.enqueue(wrapper);
   }

   TEST(thread_pool_test, run_to_completion_one_thread_one_element) {

      int array[1];
      memset(array, 0, sizeof(int) * 1);

      poly::thread_pool pool(1);
      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 1;
         });
      }

      pool.synchronize();
      
      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }
   }

   TEST(thread_pool_test, run_to_completion_one_thread_one_element_twice) {

      int array[1];
      memset(array, 0, sizeof(int) * 1);

      poly::thread_pool pool(1);
      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 1;
         });
      }

      pool.synchronize();
      
      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }

      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 0;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 0);
      }
   }

   TEST(thread_pool_test, run_to_completion_one_thread_many_elements) {

      int array[16];
      memset(array, 0, sizeof(int) * 16);

      poly::thread_pool pool(1);
      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 1;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }
   }

   TEST(thread_pool_test, run_to_completion_one_thread_many_elements_twice) {

      int array[16];
      memset(array, 0, sizeof(int) * 16);

      poly::thread_pool pool(1);
      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 1;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }

      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 0;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 0);
      }
   }


   TEST(thread_pool_test, run_to_completion_many_threads_one_element) {

      int array[1];
      memset(array, 0, sizeof(int) * 1);

      poly::thread_pool pool(2);
      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 1;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }
   }
   
   TEST(thread_pool_test, run_to_completion_many_threads_one_element_twice) {

      int array[1];
      memset(array, 0, sizeof(int) * 1);

      poly::thread_pool pool(2);
      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 1;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }

      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 0;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 0);
      }
   }

   TEST(thread_pool_test, run_to_completion_many_threads_many_elements) {

      int array[16];
      memset(array, 0, sizeof(int) * 1);

      poly::thread_pool pool(4);
      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 1;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }
   }


   TEST(thread_pool_test, run_to_completion_many_threads_many_elements_twice) {

      int array[16];
      memset(array, 0, sizeof(int) * 16);

      poly::thread_pool pool(4);
      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 1;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 1);
      }

      for (int & i : array) {
         pool.enqueue([&array, &i] {
            usleep(100000);
            i = 0;
         });
      }

      pool.synchronize();

      for (int & i : array) {
         EXPECT_EQ(i, 0);
      }
   }
   
   
}
