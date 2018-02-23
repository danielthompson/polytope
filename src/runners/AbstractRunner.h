//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_ABSTRACTRUNNER_H
#define POLYTOPE_ABSTRACTRUNNER_H

#include <memory>
#include "../samplers/AbstractSampler.h"

namespace Polytope {

   class AbstractRunner {
   public:

      // constructors

      explicit AbstractRunner(AbstractSampler *sampler)
            : Sampler(sampler) { }

      // methods

      // data
      AbstractSampler *Sampler;

   protected:
      void Trace(int x, int y);

   };

}

#endif //POLYTOPE_ABSTRACTRUNNER_H
