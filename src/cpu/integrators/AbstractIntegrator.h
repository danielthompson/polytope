//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLY_ABSTRACTINTEGRATOR_H
#define POLY_ABSTRACTINTEGRATOR_H

#include "../scenes/Scene.h"
#include "../structures/Sample.h"

namespace poly {

   class AbstractIntegrator {
   public:
      poly::Scene *Scene;
      int MaxDepth;

      AbstractIntegrator(poly::Scene *scene, int maxDepth)
         : Scene(scene), MaxDepth(maxDepth) { };

      AbstractIntegrator(int maxDepth)
         : MaxDepth(maxDepth) { };

      virtual ~AbstractIntegrator() { }

      virtual Sample GetSample(Ray &ray, int depth, int x, int y) = 0;
   };
}

#endif //POLY_ABSTRACTINTEGRATOR_H
