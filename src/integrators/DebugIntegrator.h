//
// Created by Daniel Thompson on 3/7/18.
//

#ifndef POLYTOPE_DEBUGINTEGRATOR_H
#define POLYTOPE_DEBUGINTEGRATOR_H

#include "AbstractIntegrator.h"

namespace Polytope {

   class DebugIntegrator : public AbstractIntegrator {
   public:
      DebugIntegrator(AbstractScene *scene, int maxDepth)
      : AbstractIntegrator(scene, maxDepth) { };

      Sample GetSample(Ray &ray, int depth, int x, int y) override;
   };

}

#endif //POLYTOPE_DEBUGINTEGRATOR_H
