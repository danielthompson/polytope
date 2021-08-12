//
// Created by Daniel Thompson on 3/7/18.
//

#ifndef POLY_DEBUGINTEGRATOR_H
#define POLY_DEBUGINTEGRATOR_H

#include "AbstractIntegrator.h"

namespace poly {

   class DebugIntegrator : public AbstractIntegrator {
   public:
      DebugIntegrator(std::shared_ptr<poly::scene> scene, int maxDepth)
      : AbstractIntegrator(scene, maxDepth) { };

      explicit DebugIntegrator(int maxDepth)
            : AbstractIntegrator(maxDepth) { }
      
      Sample GetSample(Ray &ray, int depth, int x, int y) override;
   };

}

#endif //POLY_DEBUGINTEGRATOR_H
