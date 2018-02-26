//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLYTOPE_SCENEBUILDER_H
#define POLYTOPE_SCENEBUILDER_H

#include "AbstractScene.h"

namespace Polytope {

   class SceneBuilder {
   public:
      static Polytope::AbstractScene* Default(float x, float y);
   };

}

#endif //POLYTOPE_SCENEBUILDER_H
