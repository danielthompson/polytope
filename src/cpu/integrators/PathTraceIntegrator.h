//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLY_PATHTRACEINTEGRATOR_H
#define POLY_PATHTRACEINTEGRATOR_H

#include "AbstractIntegrator.h"

namespace poly {

   class PathTraceIntegrator : public AbstractIntegrator {
   public:
      PathTraceIntegrator(poly::Scene *scene, int maxDepth)
            : AbstractIntegrator(scene, maxDepth) { }
      explicit PathTraceIntegrator(int maxDepth)
         : AbstractIntegrator(maxDepth) { }

      Sample GetSample(Ray &ray, int depth, int x, int y) override;

   };

}

#endif //POLY_PATHTRACEINTEGRATOR_H
