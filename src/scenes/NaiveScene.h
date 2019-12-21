//
// Created by dthompson on 23 Feb 18.
//

#ifndef POLYTOPE_NAIVESCENE_H
#define POLYTOPE_NAIVESCENE_H

#include "AbstractScene.h"

namespace Polytope {

   class NaiveScene : public AbstractScene {
   public:

      // constructors
      explicit NaiveScene(std::unique_ptr<AbstractCamera> camera) : AbstractScene(std::move(camera)) {
         ImplementationType = "Naive _scene";
      }

      // methods
      Intersection GetNearestShape(Ray &ray, int x, int y) override {
         return GetNearestShapeIteratively(Shapes, ray);
      }

      void Compile() override { }

      // destructors
      virtual ~NaiveScene() { }

      // data


   };

}


#endif //POLYTOPE_NAIVESCENE_H
