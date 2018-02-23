//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_ABSTRACTSCENE_H
#define POLYTOPE_ABSTRACTSCENE_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "../cameras/AbstractCamera.h"
#include "../shapes/AbstractShape.h"
#include "../structures/Intersection.h"

namespace Polytope {

   class AbstractScene {
   public:

      // constructors

      explicit AbstractScene(std::shared_ptr<AbstractCamera> camera) : Camera(std::move(camera)) {} ;

      // methods

      virtual Intersection GetNearestShape(Ray ray, int x, int y) = 0;

      // destructors

      virtual ~AbstractScene() { }

      // data

      std::shared_ptr<AbstractCamera> Camera;
      std::string ImplementationType = "Base Scene";
      std::vector<std::shared_ptr<AbstractShape>> Shapes;

   protected:
      Intersection GetNearestShapeIteratively(std::vector<std::shared_ptr<AbstractShape>> shapes, Ray ray);

   };

}


#endif //POLYTOPE_ABSTRACTSCENE_H
