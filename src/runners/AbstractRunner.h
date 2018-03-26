//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_ABSTRACTRUNNER_H
#define POLYTOPE_ABSTRACTRUNNER_H

#include <memory>
#include <thread>
#include "../samplers/AbstractSampler.h"
#include "../scenes/AbstractScene.h"
#include "../integrators/AbstractIntegrator.h"
#include "../films/AbstractFilm.h"

namespace Polytope {

   class AbstractRunner {
   public:

      // constructors

      explicit AbstractRunner(
            std::unique_ptr<AbstractSampler> sampler,
            AbstractScene *scene,
            std::unique_ptr<AbstractIntegrator> integrator,
            std::unique_ptr<AbstractFilm> film,
            const unsigned int numSamples,
            const Polytope::Bounds bounds)
            : Sampler(std::move(sampler)),
              Scene(scene),
              Integrator(std::move(integrator)),
              Film(std::move(film)),
              NumSamples(numSamples),
              Bounds(bounds) { }

      // methods
      virtual void Run() = 0;

      void Output();

      std::thread Spawn() {
         return std::thread(&AbstractRunner::Run, this);
      }

      // destructors
      virtual ~AbstractRunner() { }

      // data
      const unsigned int NumSamples;
      std::unique_ptr<AbstractSampler> Sampler;

   protected:

      // methods

      void Trace(int x, int y);

      // data
      AbstractScene *Scene;
      std::unique_ptr<AbstractIntegrator> Integrator;
      std::unique_ptr<AbstractFilm> Film;
      const Polytope::Bounds Bounds;
   };

}

#endif //POLYTOPE_ABSTRACTRUNNER_H
