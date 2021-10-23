//
// Created by daniel on 10/23/21.
//

#ifndef POLYTOPE_RANDOM_NUMBER_GENERATOR_H
#define POLYTOPE_RANDOM_NUMBER_GENERATOR_H

#include <random>
#include "../../../lib/pcg/pcg_random.hpp"

namespace poly {
   class random_number_generator {

      pcg32_fast generator;
      std::uniform_real_distribution<float> distribution;

      typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;
      pcg32_random_t gen = { 0, 0 };
      
   public:
      unsigned int next_uint();
      
      float next_float();
      
      float next_float_c();
      
      float next_float_ld();

      random_number_generator();

      float next_float_twiddle();
   };
}



#endif //POLYTOPE_RANDOM_NUMBER_GENERATOR_H
