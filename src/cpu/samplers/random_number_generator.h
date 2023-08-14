//
// Created by daniel on 10/23/21.
//

#ifndef POLYTOPE_RANDOM_NUMBER_GENERATOR_H
#define POLYTOPE_RANDOM_NUMBER_GENERATOR_H

namespace poly {
   // based on https://www.pcg-random.org/using-pcg-c-basic.html
   class random_number_generator {

   public:
      explicit random_number_generator(unsigned long stream_index = 0) 
        : 
          // no magic here, just a randomish starting offset
          state(0x853c49e6748fea9bULL), 
          stream_index(0xda3e39cb94b95bdbULL + (stream_index << 1ul) | 1ul)
          { }
      unsigned int next_uint();
      
      unsigned int next_uint(unsigned int floor, unsigned int ceiling);
      
      float next_float();
      
      void increment_stream(const unsigned long offset) {
         this->stream_index += offset;
      }
      
      void set_stream_index(const unsigned long stream_index_p) {
         this->stream_index = ((stream_index_p << 1ul) | 1ul);
      }

      unsigned long state;
      unsigned long stream_index;
      
   };
}



#endif //POLYTOPE_RANDOM_NUMBER_GENERATOR_H
