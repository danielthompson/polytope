//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLY_SCENE_H
#define POLY_SCENE_H

#include <memory>
#include <vector>
#include "../cameras/abstract_camera.h"
#include "../structures/intersection.h"
#include "../lights/AbstractLight.h"
#include "skyboxes/AbstractSkybox.h"
#include "../acceleration/bvh.h"
#include "../../common/utilities/Common.h"

namespace poly {
   class scene {
   public:
      
      explicit scene(std::unique_ptr<poly::abstract_camera> camera)
            : Camera(std::move(camera)), Shapes(0), Lights(0), mesh_geometry_count(0) {} ;

      ~scene() {
         for (auto element : Shapes) {
            delete element;
         }
      }

      poly::intersection intersect(poly::ray &ray, int x, int y) {
         poly::intersection intersection = bvh_root.intersect_compact(ray, x, y);
         return intersection;
      }
      
      std::unique_ptr<poly::abstract_camera> Camera { };
      std::unique_ptr<poly::AbstractSkybox> Skybox;
      
      std::vector<poly::Mesh*> Shapes;
      std::vector<std::shared_ptr<poly::texture>> textures;
      std::vector<poly::Mesh*> Lights;

      poly::bvh bvh_root;

      unsigned int mesh_geometry_count;
   };
}

#endif //POLY_SCENE_H
