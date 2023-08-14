//
// Created by daniel on 10/23/21.
//

#include "gtest/gtest.h"
#include "../src/cpu/samplers/random_number_generator.h"

namespace Tests {
   
   TEST(rng, hist_uint) {
      poly::random_number_generator g;
      std::cout << "generator state size in bytes: " << sizeof(poly::random_number_generator) << std::endl;
      
      constexpr int bits = 32;
      constexpr int bucket_count = bits;
      
      std::vector<unsigned long> buckets(bucket_count, 0);

      constexpr unsigned long trials = 1000000;
      constexpr unsigned long trial_increment = trials / 100;

      for (unsigned long i = 0; i < trials; i++) {
         unsigned int f = g.next_uint();
         for (int i = 0; i < 32; i++) {
            unsigned char bit = 0x1 & f;
            buckets[i] += bit;
            f >>= 1;
         }
      }
      for (int i = 0; i < bucket_count; i++) {
         std::cout << i << ":\t";
         for (unsigned long j = 0; j < buckets[i]; j += trial_increment) {
            std::cout << "x";
         }
         std::cout << std::endl;
      }
   }
   
   TEST(rng, hist_float) {
      poly::random_number_generator g;
      std::cout << "generator state size in bytes: " << sizeof(poly::random_number_generator) << std::endl;
      std::vector<unsigned long> buckets = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      constexpr unsigned long trials = 1000000;

      for (unsigned long i = 0; i < trials; i++) {
         float f = g.next_float();
         int bucket_index = (int)(f * 10);
         buckets[bucket_index]++;
         //std::cout << f << std::endl;
         ASSERT_TRUE(f >= 0.f);
         ASSERT_TRUE(f <= 1.f);
      }
      for (int i = 0; i < 10; i++) {
         std::cout << i << ": ";
         for (unsigned long j = 0; j < buckets[i]; j += (trials / 100)) {
            std::cout << "x";
         }
         std::cout << std::endl;
      }
   }
}
