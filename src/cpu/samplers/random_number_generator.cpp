//
// Created by daniel on 10/23/21.
//

#include <cassert>
#include "random_number_generator.h"


namespace poly {
   
   float random_number_generator::next_float() {
      // https://github.com/wjakob/pcg32/
      union {
         unsigned int u;
         float f;
      } x;
      x.u = (next_uint() >> 9) | 0x3f800000u;
      return x.f - 1.0f;
      
   }

   unsigned int random_number_generator::next_uint() {
      unsigned long oldstate = state;
      state = oldstate * 0x5851f42d4c957f2dULL + stream_index;
      unsigned int xorshifted = (unsigned int)(((oldstate >> 18u) ^ oldstate) >> 27u);
      unsigned int rot = (unsigned int)(oldstate >> 59u);
      return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
   }

   unsigned int random_number_generator::next_uint(unsigned int floor, unsigned int ceiling) {
      assert(ceiling > floor);
      const unsigned range = ceiling - floor;
      unsigned int next = floor + (next_uint() % range);
      assert(next >= floor);
      assert(next < ceiling);
      
      return next;
      
   }
}

