//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_PIXELRUNNER_H
#define POLYTOPE_PIXELRUNNER_H

#include "AbstractRunner.h"

namespace Polytope {

   class PixelRunner : AbstractRunner {
   public:
      PixelRunner(AbstractSampler *sampler, int x, int y);

      void Run();
   private:
      int _x, _y;
   };

}


#endif //POLYTOPE_PIXELRUNNER_H
