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
#include "../lights/AbstractLight.h"

namespace Polytope {

   class AbstractScene {
   public:

      // constructors

      explicit AbstractScene(std::unique_ptr<AbstractCamera> camera)
            : Camera(std::move(camera)) {} ;

      // methods

      virtual Intersection GetNearestShape(Ray &ray, int x, int y) = 0;
      virtual void Compile() = 0;

      // destructors

      virtual ~AbstractScene() = default;

      // data

      std::unique_ptr<AbstractCamera> Camera{};
      std::string ImplementationType = "Base Scene";
      std::vector<AbstractShape*> Shapes{};
      std::vector<AbstractLight*> Lights{};

   protected:
      Intersection GetNearestShapeIteratively(std::vector<AbstractShape*> &shapes, Ray &ray) const;

   };

}


#endif //POLYTOPE_ABSTRACTSCENE_H
