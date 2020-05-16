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

      PixelRunner(
            std::unique_ptr<AbstractSampler> sampler,
            AbstractScene *scene,
            std::unique_ptr<AbstractIntegrator> integrator,
            std::unique_ptr<AbstractFilm> film,
            const Polytope::Bounds bounds,
            const unsigned int numSamples)
            : AbstractRunner(
               std::move(sampler),
               scene,
               std::move(integrator),
               std::move(film),
               numSamples,
               bounds
            ) { }


      // methods


      void Run(int threadId) override;

   };

}


#endif //POLYTOPE_PIXELRUNNER_H
