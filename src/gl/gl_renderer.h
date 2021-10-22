//
// Created by daniel on 1/8/20.
//

#ifndef POLY_GLRENDERER_H
#define POLY_GLRENDERER_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "../cpu/runners/runner.h"
#include "../cpu/integrators/PathTraceIntegrator.h"
#include "bb_vao.h"

namespace poly {
   class gl_renderer {
   public:
      explicit gl_renderer(std::shared_ptr<poly::runner> source_scene);
      poly::bb_vao bb;
      std::shared_ptr<poly::runner> runner;
      GLFWwindow* window;
      void render();
   };
}



#endif //POLY_GLRENDERER_H