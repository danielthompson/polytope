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
      explicit NaiveScene(const std::shared_ptr<Polytope::AbstractCamera> &camera) : AbstractScene(camera) {}

      Intersection GetNearestShape(Ray ray, int x, int y) override {
         return GetNearestShapeIteratively(Shapes, ray);
      }

      // destructors
      virtual ~NaiveScene() { }

   };

}


#endif //POLYTOPE_NAIVESCENE_H
