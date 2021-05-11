//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLY_PERSPECTIVECAMERA_H
#define POLY_PERSPECTIVECAMERA_H

#include "AbstractCamera.h"
#include "../../common/structures/Point2.h"

namespace poly {
   class PerspectiveCamera : public AbstractCamera {
   public:
      PerspectiveCamera(const CameraSettings &settings, const Transform &cameraToWorld, bool leftHanded);

      Ray get_ray_for_pixel(const Point2f pixel) override;
      Ray get_ray_for_ndc(const Point2f ndc) override;

   public:
      float OneOverWidth;
      float OneOverHeight;
      float AspectRatio;
      float TanFOVOver2;
      bool LeftHanded = false;
   };
}

#endif //POLY_PERSPECTIVECAMERA_H
