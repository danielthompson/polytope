//
// Created by daniel on 5/15/20.
//

#ifndef POLYTOPE_PATH_TRACER_CUH
#define POLYTOPE_PATH_TRACER_CUH

#include <utility>

#include "../gpu_memory_manager.h"

namespace Polytope {
   class PathTracerKernel {
   public:
      explicit PathTracerKernel(Polytope::GPUMemoryManager* memory_manager) 
         : memory_manager(memory_manager) { }
         
      void Trace();

      Polytope::GPUMemoryManager* memory_manager;
   };
}

#endif //POLYTOPE_PATH_TRACER_CUH
