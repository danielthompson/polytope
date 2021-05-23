//
// Created by daniel on 1/8/20.
//

#ifndef POLY_GLRENDERER_H
#define POLY_GLRENDERER_H

#include "../cpu/scenes/scene.h"

namespace poly {
   class gl_renderer {
   public:
      void render(std::shared_ptr<poly::scene> scene_p);
   };
}



#endif //POLY_GLRENDERER_H
