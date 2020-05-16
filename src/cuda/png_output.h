//
// Created by daniel on 5/16/20.
//

#ifndef POLYTOPE_PNG_OUTPUT_H
#define POLYTOPE_PNG_OUTPUT_H

#include "gpu_memory_manager.h"

namespace Polytope {
   class OutputPNG {
   public:
      static void Output(const std::shared_ptr<Polytope::GPUMemoryManager>& memory_manager);
   };
}

#endif //POLYTOPE_PNG_OUTPUT_H
