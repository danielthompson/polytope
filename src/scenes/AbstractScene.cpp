//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractScene.h"
#include "../Constants.h"
#include <cmath>

namespace Polytope {

   Intersection AbstractScene::GetNearestShapeIteratively(std::vector<AbstractMesh*> &shapes, Ray &ray) const {

      Intersection intersection;

      for (auto shape : shapes) {
         if (shape->BoundingBox &&!shape->BoundingBox->Hits(ray)) {
            continue;
         }

         shape->intersect(ray, &intersection);
      }

      return intersection;
   }
}
