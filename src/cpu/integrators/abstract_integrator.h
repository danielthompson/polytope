//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLY_ABSTRACTINTEGRATOR_H
#define POLY_ABSTRACTINTEGRATOR_H

#include "../scenes/scene.h"
#include "../structures/Sample.h"

namespace poly {

   class abstract_integrator {
   public:
      std::shared_ptr<poly::scene> Scene;
      int MaxDepth;

      abstract_integrator(std::shared_ptr<poly::scene> scene, int max_depth)
         : Scene(scene), MaxDepth(max_depth) { };

      explicit abstract_integrator(int max_depth)
         : MaxDepth(max_depth) { };

      virtual ~abstract_integrator() {
         int base = 0;
      };

      virtual poly::Sample get_sample(poly::ray &ray, int depth, int x, int y) = 0;
   };
}

#endif //POLY_ABSTRACTINTEGRATOR_H
