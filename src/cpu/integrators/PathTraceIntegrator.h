//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLY_PATHTRACEINTEGRATOR_H
#define POLY_PATHTRACEINTEGRATOR_H

#include "abstract_integrator.h"

namespace poly {

   class PathTraceIntegrator : public abstract_integrator {
   public:
      PathTraceIntegrator(std::shared_ptr<poly::scene> scene, int max_depth)
            : abstract_integrator(scene, max_depth) { }
      explicit PathTraceIntegrator(int max_depth)
         : abstract_integrator(max_depth) { }

      ~PathTraceIntegrator() override {
         int i = 0;
      }

      poly::Sample get_sample(poly::ray &ray, int depth, int x, int y) override;

   };

}

#endif //POLY_PATHTRACEINTEGRATOR_H
