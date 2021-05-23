//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLY_SCENE_H
#define POLY_SCENE_H

#include <memory>
#include <vector>
#include "../cameras/AbstractCamera.h"
#include "../structures/Intersection.h"
#include "../lights/AbstractLight.h"
#include "skyboxes/AbstractSkybox.h"
#include "../acceleration/bvh.h"
#include "../../common/utilities/Common.h"

namespace poly {
   class scene {
   public:
      
      explicit scene(std::unique_ptr<poly::AbstractCamera> camera)
            : Camera(std::move(camera)), Shapes(0), Lights(0), mesh_geometry_count(0) {} ;

      ~scene() {
         for (auto element : Shapes) {
            delete element;
         }
      }
      
      poly::Intersection intersect(poly::Ray &ray, int x, int y) {
         poly::Intersection bvh_intersection;
         bvh_intersection.x = x;
         bvh_intersection.y = y;
         bvh_root.intersect_compact(ray, bvh_intersection);
         return bvh_intersection;
      }
      
      std::unique_ptr<poly::AbstractCamera> Camera { };
      std::unique_ptr<poly::AbstractSkybox> Skybox;
      
      std::vector<poly::Mesh*> Shapes;
      std::vector<std::shared_ptr<poly::texture>> textures;
      std::vector<poly::Mesh*> Lights;

      poly::bvh bvh_root;

      unsigned int mesh_geometry_count;
   };
}

#endif //POLY_SCENE_H
