//
// Created by Daniel on 19-Mar-18.
//

#ifndef POLYTOPE_TILERUNNER_H
#define POLYTOPE_TILERUNNER_H

#include <mutex>
#include "AbstractRunner.h"
#include "../utilities/Logger.h"

namespace Polytope {

   class TileRunner : public AbstractRunner {
   public:
      explicit TileRunner(
            std::unique_ptr<AbstractSampler> sampler,
            AbstractScene *scene,
            std::unique_ptr<AbstractIntegrator> integrator,
            std::unique_ptr<AbstractFilm> film,
            Polytope::Bounds bounds,
            unsigned int numSamples);

      void Run() override;
   };
}

#endif //POLYTOPE_TILERUNNER_H
