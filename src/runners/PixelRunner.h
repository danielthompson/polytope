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

      PixelRunner(AbstractSampler *sampler, AbstractScene *scene, AbstractIntegrator *integrator, AbstractFilm *film,
                        const Polytope::Bounds bounds, const unsigned int numSamples)
            : AbstractRunner(sampler, scene, integrator, film, numSamples, bounds) { }

      // methods


      void Run();

   };

}


#endif //POLYTOPE_PIXELRUNNER_H
