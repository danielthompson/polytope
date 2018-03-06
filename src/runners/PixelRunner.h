//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_PIXELRUNNER_H
#define POLYTOPE_PIXELRUNNER_H

#include "AbstractRunner.h"
#include "../integrators/AbstractIntegrator.h"
#include "../films/AbstractFilm.h"

namespace Polytope {

   class PixelRunner : public AbstractRunner {
   public:

      // constructors

      PixelRunner(AbstractSampler *sampler, AbstractScene *scene, AbstractIntegrator *integrator,
                  AbstractFilm *film, int x, int y)
            : AbstractRunner(sampler, scene, integrator, film), _x(x), _y(y) { }

      // methods


      void Run();
   private:
      int _x, _y;


   };

}


#endif //POLYTOPE_PIXELRUNNER_H
