//
// Created by daniel on 5/15/20.
//

#ifndef POLY_PATH_TRACER_CUH
#define POLY_PATH_TRACER_CUH

#include <utility>

#include "../context.h"

namespace poly {
   class path_tracer {
   public:
      explicit path_tracer(poly::device_context* device_context) 
         : device_context(device_context) { }
         
      void Trace(unsigned int num_samples) const;

      bool unit_test_hit_ray_against_bounding_box(const poly::ray &ray, const float* const device_aabb);
      
      poly::device_context* device_context;
   };
}

#endif //POLY_PATH_TRACER_CUH
