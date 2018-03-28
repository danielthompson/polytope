//
// Created by Daniel on 21-Mar-18.
//

#ifndef POLYTOPE_TRACER_H
#define POLYTOPE_TRACER_H

#include "utilities/Logger.h"
#include "scenes/AbstractScene.h"
#include "runners/AbstractRunner.h"

namespace Polytope {

   class Tracer {
   public:

      explicit Tracer(Polytope::Logger logger) : Logger(logger) { }

      // methods
      void Run();
      void Compile(AbstractScene *scene);

      // data
      Polytope::Logger Logger;

   };

}


#endif //POLYTOPE_TRACER_H
