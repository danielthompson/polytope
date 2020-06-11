//
// Created by daniel on 5/16/20.
//

#ifndef POLY_PNG_OUTPUT_H
#define POLY_PNG_OUTPUT_H

#include "gpu_memory_manager.h"

namespace poly {
   class OutputPNG {
   public:
      static void Output(const poly::GPUMemoryManager* memory_manager);
   };
}

#endif //POLY_PNG_OUTPUT_H
