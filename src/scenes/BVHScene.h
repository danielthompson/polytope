//
// Created by dthompson on 23 Feb 18.
//

#ifndef POLYTOPE_BVHSCENE_H
#define POLYTOPE_BVHSCENE_H

#include "AbstractScene.h"

namespace Polytope {

   class BVHScene : public AbstractScene {
   public:

      explicit BVHScene(std::unique_ptr<AbstractCamera> camera) : AbstractScene(std::move(camera)) {
         ImplementationType = "BVH Scene";
      }
      ~BVHScene() override = default;

      Intersection GetNearestShape(Ray &ray, int x, int y) override;
      void Compile() override;
   };
}

#endif //POLYTOPE_BVHSCENE_H
