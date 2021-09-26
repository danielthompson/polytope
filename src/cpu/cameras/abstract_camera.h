//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLY_ABSTRACTCAMERA_H
#define POLY_ABSTRACTCAMERA_H

#include "../structures/Vectors.h"
#include "../../common/structures/point2.h"
#include "../structures/transform.h"
#include "camera_settings.h"

namespace poly {

   class abstract_camera {
   public:

      abstract_camera(const poly::camera_settings &settings, const poly::transform &cameraToWorld)
            : settings(settings), camera_to_world(cameraToWorld) { }
      virtual ~abstract_camera() = default;

      virtual poly::ray get_ray_for_pixel(poly::point2f pixel) = 0;
      virtual poly::ray get_ray_for_ndc(poly::point2f pixel) = 0;

      const poly::camera_settings settings;
      const poly::transform camera_to_world;

      poly::point eye;
      poly::point lookAt;
      poly::vector up;

   protected:
      const poly::vector DefaultDirection = vector(0, 0, -1);
      const poly::point DefaultOrigin = point(0, 0, 0);
   };
}


#endif //POLY_ABSTRACTCAMERA_H
