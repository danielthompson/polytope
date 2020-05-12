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
#include "../shapes/abstract_mesh.h"
#include "../structures/Intersection.h"
#include "../lights/AbstractLight.h"
#include "skyboxes/AbstractSkybox.h"

namespace Polytope {

   class AbstractScene {
   public:

      // constructors

      explicit AbstractScene(std::unique_ptr<AbstractCamera> camera)
            : Camera(std::move(camera)), Shapes(0), Lights(0) {} ;

      // methods

      virtual Intersection GetNearestShape(Ray &ray, int x, int y) = 0;
      virtual void Compile() { };

      // destructors

      virtual ~AbstractScene() = default;

      // data

      std::unique_ptr<AbstractCamera> Camera{};
      std::string ImplementationType = "Base Scene";
      std::vector<AbstractMesh*> Shapes;
      std::vector<AbstractMesh*> Lights;
      std::unique_ptr<AbstractSkybox> Skybox;

   protected:
      Intersection GetNearestShapeIteratively(std::vector<AbstractMesh*> &shapes, Ray &ray) const;
   };

}


#endif //POLYTOPE_ABSTRACTSCENE_H
