//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLY_ABSTRACTINTEGRATOR_H
#define POLY_ABSTRACTINTEGRATOR_H

#include "../scenes/scene.h"
#include "../structures/Sample.h"

namespace poly {

   class AbstractIntegrator {
   public:
      std::shared_ptr<poly::scene> Scene;
      int MaxDepth;

      AbstractIntegrator(poly::scene *scene, int maxDepth)
         : Scene(scene), MaxDepth(maxDepth) { };

      AbstractIntegrator(int maxDepth)
         : MaxDepth(maxDepth) { };

      virtual ~AbstractIntegrator() { }

      virtual Sample GetSample(Ray &ray, int depth, int x, int y) = 0;
   };
}

#endif //POLY_ABSTRACTINTEGRATOR_H
