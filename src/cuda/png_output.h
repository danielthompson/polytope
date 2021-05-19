//
// Created by daniel on 5/16/20.
//

#ifndef POLY_PNG_OUTPUT_H
#define POLY_PNG_OUTPUT_H

#include "context.h"

namespace poly {
   class output_png {
   public:
      static void output(const poly::render_context *render_context, const std::string& filename);
   };
}

#endif //POLY_PNG_OUTPUT_H
