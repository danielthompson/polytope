//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLYTOPE_ABSTRACTINTEGRATOR_H
#define POLYTOPE_ABSTRACTINTEGRATOR_H

#include "../scenes/AbstractScene.h"
#include "../structures/Sample.h"

namespace Polytope {

   class AbstractIntegrator {
   public:
      AbstractScene *Scene;
      int MaxDepth;

      AbstractIntegrator(AbstractScene *scene, int maxDepth)
         : Scene(scene), MaxDepth(maxDepth) { };

      AbstractIntegrator(int maxDepth)
         : MaxDepth(maxDepth) { };

      virtual ~AbstractIntegrator() { }

      virtual Sample GetSample(Ray &ray, int depth, int x, int y) = 0;
   };
}

#endif //POLYTOPE_ABSTRACTINTEGRATOR_H
