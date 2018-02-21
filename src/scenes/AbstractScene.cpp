//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractScene.h"
#include "../Constants.h"
#include <cmath>

namespace Polytope {


   Intersection AbstractScene::GetNearestShapeIteratively(std::vector<AbstractShape> shapes, Ray ray) {
      int nearestShapeIndex = -1;

      float closestT = ray.MinT;

      bool test = false;

      for (int i = 0; i < shapes.size(); i++) {

         AbstractShape *shape = &shapes[i];

         bool hits = shape->Hits(ray);

         test = (hits && ray.MinT >= Epsilon && ray.MinT < closestT);

         nearestShapeIndex = test ? i : nearestShapeIndex;

         closestT = test ? ray.MinT : closestT;
      }

      Intersection closestStateToRay;

      if (nearestShapeIndex >= 0) {
         AbstractShape *nearestShape = &shapes[nearestShapeIndex];

         closestStateToRay = nearestShape->Intersect(ray);

         if (isnan(closestStateToRay.Location.x)) {
            // wtf?
            closestStateToRay = nearestShape->Intersect(ray);
         }
      }

      return closestStateToRay;
   }
}