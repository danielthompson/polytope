//
// Created by Daniel on 19-Mar-18.
//

#ifndef POLY_TILERUNNER_H
#define POLY_TILERUNNER_H

#include <mutex>
#include "AbstractRunner.h"
#include "../../common/utilities/Logger.h"

namespace poly {

   class TileRunner : public AbstractRunner {
   public:
      explicit TileRunner(
            std::unique_ptr<AbstractSampler> sampler,
            std::shared_ptr<poly::scene> scene,
            std::unique_ptr<AbstractIntegrator> integrator,
            std::unique_ptr<AbstractFilm> film,
            poly::Bounds bounds,
            unsigned int numSamples);

      void Run(int threadId) override;
   };
}

#endif //POLY_TILERUNNER_H
