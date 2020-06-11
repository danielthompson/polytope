//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLY_SCENE_H
#define POLY_SCENE_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "../cameras/AbstractCamera.h"
#include "../shapes/abstract_mesh.h"
#include "../structures/Intersection.h"
#include "../lights/AbstractLight.h"
#include "skyboxes/AbstractSkybox.h"
#include "../acceleration/bvh.h"
#include "../../common/utilities/Common.h"

namespace poly {
   class Scene {
   public:
      
      explicit Scene(std::unique_ptr<AbstractCamera> camera)
            : Camera(std::move(camera)), Shapes(0), Lights(0) {} ;
      virtual ~Scene() = default;

      Intersection GetNearestShape(Ray &ray, int x, int y);
      
      Intersection GetNearestShapeIteratively(std::vector<TMesh*> &shapes, Ray &ray) const;

      std::unique_ptr<AbstractCamera> Camera{};
      std::vector<TMesh*> Shapes;
      std::vector<TMesh*> Lights;
      std::unique_ptr<AbstractSkybox> Skybox;
      
      bvh bvh_root;
   };
}

#endif //POLY_SCENE_H
