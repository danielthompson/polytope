//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractScene.h"
#include "../Constants.h"
#include <cmath>

namespace Polytope {

   Intersection AbstractScene::GetNearestShapeIteratively(std::vector<AbstractShape*> &shapes, Ray &ray) const {
      int nearestShapeIndex = -1;

      float closestT = ray.MinT;

      bool test = false;

      for (int i = 0; i < shapes.size(); i++) {

         AbstractShape *shape = shapes[i];

         bool hits = shape->Hits(ray);

         test = (hits && ray.MinT >= Epsilon && ray.MinT < closestT);

         nearestShapeIndex = test ? i : nearestShapeIndex;

         closestT = test ? ray.MinT : closestT;
      }

      Intersection intersection;

      if (nearestShapeIndex >= 0) {
         AbstractShape *nearestShape = shapes[nearestShapeIndex];

         nearestShape->Intersect(ray, &intersection);

         intersection.Hits = true;
         intersection.Shape = nearestShape;

         if (std::isnan(intersection.Location.x)) {
            // wtf?
            nearestShape->Intersect(ray, &intersection);
         }
      }

      return intersection;
   }
}