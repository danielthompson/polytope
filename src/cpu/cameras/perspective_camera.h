//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLY_PERSPECTIVECAMERA_H
#define POLY_PERSPECTIVECAMERA_H

#include "abstract_camera.h"
#include "../../common/structures/point2.h"

namespace poly {
   class perspective_camera : public abstract_camera {
   public:
      perspective_camera(const camera_settings &settings, const transform &camera_to_world, bool left_handed);

      poly::ray get_ray_for_pixel(point2f pixel) override;
      poly::ray get_ray_for_ndc(point2f ndc) override;

   public:
      float one_over_width;
      float one_over_height;
      float aspect_ratio;
      float tan_fov_over_2;
      bool left_handed = false;
   };
}

#endif //POLY_PERSPECTIVECAMERA_H
