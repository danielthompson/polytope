//
// Created by Daniel on 21-Mar-18.
//

#ifndef POLYTOPE_TRACER_H
#define POLYTOPE_TRACER_H

#include "scenes/AbstractScene.h"
#include "runners/AbstractRunner.h"
#include "utilities/Options.h"

namespace Polytope {

   class Tracer {
   public:

      Tracer(Polytope::Options options)
            : Options(options) { }

      // methods
      void Run();
      void Compile(AbstractScene *scene);

      // data
      Polytope::Options Options;

   };

}


#endif //POLYTOPE_TRACER_H
