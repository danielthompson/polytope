//
// Created by daniel on 1/8/20.
//

#ifndef POLYTOPE_GLRENDERER_H
#define POLYTOPE_GLRENDERER_H

#include "../scenes/AbstractScene.h"

namespace Polytope {
   class GLRenderer {
   public:
      void Render(Polytope::AbstractScene* scene);
   };
}



#endif //POLYTOPE_GLRENDERER_H
