//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_ABSTRACTRUNNER_H
#define POLYTOPE_ABSTRACTRUNNER_H

#include <memory>
#include "../samplers/AbstractSampler.h"
#include "../scenes/AbstractScene.h"
#include "../integrators/AbstractIntegrator.h"
#include "../films/AbstractFilm.h"

namespace Polytope {

   class AbstractRunner {
   public:

      // constructors

      explicit AbstractRunner(AbstractSampler *sampler, AbstractScene *scene, AbstractIntegrator *integrator,
                              AbstractFilm *film)
            : Sampler(sampler), Scene(scene), Integrator(integrator), Film(film){ }

      // methods
      virtual void Run() = 0;

      // destructors
      virtual ~AbstractRunner() { }

      // data
      AbstractSampler *Sampler;

   protected:

      // methods

      void Trace(int x, int y);

      // data

      AbstractScene *Scene;
      AbstractIntegrator *Integrator;
      AbstractFilm *Film;
   };

}

#endif //POLYTOPE_ABSTRACTRUNNER_H
