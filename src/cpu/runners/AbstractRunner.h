//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_ABSTRACTRUNNER_H
#define POLY_ABSTRACTRUNNER_H

#include <memory>
#include <thread>
#include "../samplers/samplers.h"
#include "../scenes/Scene.h"
#include "../integrators/AbstractIntegrator.h"
#include "../films/AbstractFilm.h"
#include "../structures/stats.h"

namespace poly {

   class AbstractRunner {
   public:

      // constructors

      explicit AbstractRunner(
            std::unique_ptr<AbstractSampler> sampler,
            Scene *scene,
            std::unique_ptr<AbstractIntegrator> integrator,
            std::unique_ptr<AbstractFilm> film,
            const unsigned int numSamples,
            const poly::Bounds bounds)
            : Sampler(std::move(sampler)),
              Scene(scene),
              Integrator(std::move(integrator)),
              Film(std::move(film)),
              NumSamples(numSamples),
              Bounds(bounds) { }

      // methods
      virtual void Run(int threadId) = 0;
      void Trace(int x, int y) const;

      void Output() const;

      std::thread Spawn(const int id) {
         return std::thread(&AbstractRunner::Run, this, id);
      }

      // destructors
      virtual ~AbstractRunner() = default;

      // data
      unsigned int NumSamples;
      std::unique_ptr<AbstractSampler> Sampler;
      std::unique_ptr<AbstractFilm> Film;
      poly::Scene *Scene;
      std::unique_ptr<AbstractIntegrator> Integrator;

      const poly::Bounds Bounds;
   };

}

#endif //POLY_ABSTRACTRUNNER_H
