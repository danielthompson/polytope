//
// Created by daniel on 10/23/21.
//


#include "random_number_generator.h"
#include "../../../lib/pcg/pcg_random.hpp"

namespace poly {
   random_number_generator::random_number_generator() {
      pcg_extras::seed_seq_from<std::random_device> seed_source;
      generator = pcg32_fast(seed_source);
      distribution = std::uniform_real_distribution<float>();
   }

   float random_number_generator::next_float() {
      return distribution(generator);
   }

   float random_number_generator::next_float_ld() {
      return (float)std::ldexp(next_uint(), -32);
   }

   // https://github.com/tensorflow/tensorflow/blob/ed6f783690a35e54441881930d4291894bc03285/tensorflow/core/lib/random/random_distributions.h#L402
   float random_number_generator::next_float_twiddle() {
      // IEEE754 floats are formatted as follows (MSB first):
      //    sign(1) exponent(8) mantissa(23)
      // Conceptually construct the following:
      //    sign == 0
      //    exponent == 127  -- an excess 127 representation of a zero exponent
      //    mantissa == 23 random bits
      const unsigned int man = next_uint() & 0x7fffffu;  // 23 bit mantissa
      const unsigned int exp = static_cast<unsigned int>(127);
      const unsigned int val = (exp << 23) | man;

      // Assumes that endian-ness is same for float and uint32.
      float result;
      std::memcpy(&result, &val, sizeof(val));
      return result - 1.0f;
   }
   
   unsigned int random_number_generator::next_uint() {
      uint64_t oldstate = gen.state;
      // Advance internal state
      gen.state = oldstate * 6364136223846793005ULL + (gen.inc|1);
      // Calculate output function (XSH RR), uses old state for max ILP
      uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
      uint32_t rot = oldstate >> 59u;

      return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
   }
   
   
}

