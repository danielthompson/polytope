//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_ABSTRACTRUNNER_H
#define POLYTOPE_ABSTRACTRUNNER_H

#include "../samplers/AbstractSampler.h"

namespace Polytope {

   class AbstractRunner {
   public:
      AbstractSampler *Sampler;

   protected:
      void Trace(int x, int y);

   };

}

#endif //POLYTOPE_ABSTRACTRUNNER_H
