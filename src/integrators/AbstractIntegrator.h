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
      // methods
      virtual Sample GetSample(Ray &ray, int depth, int x, int y) = 0;

      // destructors
      virtual ~AbstractIntegrator() { }
   protected:

      // constructors
      AbstractIntegrator(AbstractScene *scene, int maxDepth)
      : Scene(scene), MaxDepth(maxDepth) { };

      // data
      AbstractScene *Scene;
      int MaxDepth;
   };

}


#endif //POLYTOPE_ABSTRACTINTEGRATOR_H
