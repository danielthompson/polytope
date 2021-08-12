//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_ABSTRACTRUNNER_H
#define POLY_ABSTRACTRUNNER_H

#include <memory>
#include <thread>
#include "../samplers/samplers.h"
#include "../scenes/scene.h"
#include "../integrators/AbstractIntegrator.h"
#include "../films/AbstractFilm.h"
#include "../structures/stats.h"

namespace poly {

   class AbstractRunner {
   public:
      AbstractRunner(
            std::unique_ptr<AbstractSampler> sampler,
            std::shared_ptr<poly::scene> scene,
            std::shared_ptr<poly::AbstractIntegrator> integrator,
            std::unique_ptr<AbstractFilm> film,
            const unsigned int numSamples,
            const poly::Bounds bounds);

      void Run(int threadId) const;
      void Trace(const int x, const int y) const;
      void Output() const {
         Film->Output();
      };
      std::thread Spawn(const int id) {
         return std::thread(&AbstractRunner::Run, this, id);
      }

      unsigned int NumSamples;
      std::unique_ptr<AbstractSampler> Sampler;
      std::unique_ptr<AbstractFilm> Film;
      std::shared_ptr<poly::scene> Scene;
      std::shared_ptr<poly::AbstractIntegrator> Integrator;

      const poly::Bounds Bounds;
   };

}

#endif //POLY_ABSTRACTRUNNER_H
