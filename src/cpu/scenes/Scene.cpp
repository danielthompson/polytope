//
// Created by Daniel on 20-Feb-18.
//

#include "Scene.h"
#include "../constants.h"
#include <cmath>

namespace poly {

   Intersection Scene::GetNearestShape(Ray &ray, int x, int y) {
      return GetNearestShapeIteratively(this->Shapes, ray);
   }

   Intersection Scene::GetNearestShapeIteratively(std::vector<TMesh*> &shapes, Ray &ray) const {

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
