//
// Created by daniel on 10/23/21.
//

#include <chrono>
#include "gtest/gtest.h"
#include "../src/cpu/samplers/random_number_generator.h"

namespace Tests {
   TEST(rng, test1) {
      poly::random_number_generator g;
      float f = g.next_float();
      std::cout << sizeof(g) << " " << f;
      ASSERT_TRUE(f >= 0.f);
      ASSERT_TRUE(f <= 1.f);
   }

   TEST(rng, test3) {
      poly::random_number_generator g;
      std::cout << sizeof(g) << std::endl;
      for (int i = 0; i < 10; i++) {
         float f = (float)((double)std::rand() / (double)RAND_MAX);
         std::cout << " " << f;
         ASSERT_TRUE(f >= 0.f);
         ASSERT_TRUE(f <= 1.f);
      }
   }

   TEST(rng, test4) {
      poly::random_number_generator g;
      std::cout << sizeof(g) << std::endl;
      for (int i = 0; i < 10; i++) {
         float f = g.next_float_ld();
         std::cout << " " << f;
         ASSERT_TRUE(f >= 0.f);
         ASSERT_TRUE(f <= 1.f);
      }
   }

   TEST(rng, test5) {
      poly::random_number_generator g;
      std::cout << sizeof(g) << std::endl;
      for (int j = 0; j < 10; j++) {
         for (int i = 0; i < 10; i++) {
            float f = g.next_float_twiddle();
            std::cout << "\t" << f;
            ASSERT_TRUE(f >= 0.f);
            ASSERT_TRUE(f <= 1.f);
         }
         std::cout << std::endl;
      }
   }

   TEST(rng, speed) {
      poly::random_number_generator g;
      
      // warm up
      float f;
      for (int i = 0; i < 100000; i++) {
         f = g.next_float();
      }
      
      std::vector<double> durations;
      
      for (int trial_index = 0; trial_index < 10; trial_index++) {
         const auto bound_start = std::chrono::system_clock::now();
         for (int i = 0; i < 1000000; i++) {
            f = g.next_float();
//            f = (float)((double)std::rand() / (double)RAND_MAX);
            
         }

         const auto bound_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> bound_duration = bound_end - bound_start;
         durations.push_back(bound_duration.count());
      }
      
      // calculate average
      double sum = 0.f;
      for (int trial_index = 0; trial_index < 10; trial_index++) {
         sum += durations[trial_index];
      }
      
      double average = sum / 10;

      std::cout << "Speed: " << average << "s.";
   }
}
