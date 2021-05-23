//
// Created by daniel on 1/8/20.
//

#ifndef POLY_GLRENDERER_H
#define POLY_GLRENDERER_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "../cpu/scenes/scene.h"
#include "bb_vao.h"

namespace poly {
   class gl_renderer {
   public:
      explicit gl_renderer(std::shared_ptr<poly::scene> source_scene);
      poly::bb_vao bb;
      std::shared_ptr<poly::scene> scene;
      GLFWwindow* window;
      void render();
   };
}



#endif //POLY_GLRENDERER_H
