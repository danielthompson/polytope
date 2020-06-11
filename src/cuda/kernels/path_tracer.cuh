//
// Created by daniel on 5/15/20.
//

#ifndef POLY_PATH_TRACER_CUH
#define POLY_PATH_TRACER_CUH

#include <utility>

#include "../gpu_memory_manager.h"

namespace poly {
   class PathTracerKernel {
   public:
      explicit PathTracerKernel(poly::GPUMemoryManager* memory_manager) 
         : memory_manager(memory_manager) { }
         
      void Trace() const;

      poly::GPUMemoryManager* memory_manager;
   };
}

#endif //POLY_PATH_TRACER_CUH
