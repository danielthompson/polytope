//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLYTOPE_PATHTRACEINTEGRATOR_H
#define POLYTOPE_PATHTRACEINTEGRATOR_H

#include "AbstractIntegrator.h"

namespace Polytope {

   class PathTraceIntegrator : public AbstractIntegrator {
   public:
      PathTraceIntegrator(AbstractScene *scene, int maxDepth)
            : AbstractIntegrator(scene, maxDepth) { }

      Sample GetSample(Ray &ray, int depth, int x, int y) override;


   };

}

#endif //POLYTOPE_PATHTRACEINTEGRATOR_H
