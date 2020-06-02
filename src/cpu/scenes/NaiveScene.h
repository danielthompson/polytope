//
// Created by dthompson on 23 Feb 18.
//

#ifndef POLYTOPE_NAIVESCENE_H
#define POLYTOPE_NAIVESCENE_H

#include "AbstractScene.h"

namespace Polytope {

   class NaiveScene : public AbstractScene {
   public:

      explicit NaiveScene(std::unique_ptr<AbstractCamera> camera) : AbstractScene(std::move(camera)) {
         ImplementationType = "Naive _scene";
      }
      ~NaiveScene() override = default;

      Intersection GetNearestShape(Ray &ray, int x, int y) override {
         bool debug = false;
         if (x == 245 && y == 64)
            debug = true;
         return GetNearestShapeIteratively(Shapes, ray);
      }
   };
}

#endif //POLYTOPE_NAIVESCENE_H
