//
// Created by daniel on 1/8/20.
//

#ifndef POLY_GLRENDERER_H
#define POLY_GLRENDERER_H

#include "../cpu/scenes/Scene.h"

namespace poly {
   class GLRenderer {
   public:
      void Render(poly::Scene* scene);
   };
}



#endif //POLY_GLRENDERER_H
