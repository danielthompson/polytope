//
// Created by Daniel Thompson on 3/7/18.
//

#ifndef POLY_DEBUGINTEGRATOR_H
#define POLY_DEBUGINTEGRATOR_H

#include "abstract_integrator.h"

namespace poly {

   class DebugIntegrator : public abstract_integrator {
   public:
      DebugIntegrator(std::shared_ptr<poly::scene> scene, int max_depth)
      : abstract_integrator(scene, max_depth) { };

      explicit DebugIntegrator(int max_depth)
            : abstract_integrator(max_depth) { }
      
      poly::Sample get_sample(poly::ray &ray, int depth, int x, int y) override;
   };

}

#endif //POLY_DEBUGINTEGRATOR_H
