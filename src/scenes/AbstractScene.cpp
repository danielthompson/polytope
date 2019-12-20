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

      unsigned int i = -1;

      for (auto shape : shapes) {
         i++;
         if (shape->BoundingBox) {
            Intersection boundingBoxIntersection;
            float priorMinT = ray.MinT;
            shape->BoundingBox->Intersect(ray, &boundingBoxIntersection);
            ray.MinT = priorMinT;
            if (!boundingBoxIntersection.Hits) {
               continue;
            }
         }
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

      }

      return intersection;
   }
}
