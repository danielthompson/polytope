//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_PERSPECTIVECAMERA_H
#define POLYTOPE_PERSPECTIVECAMERA_H

#include "AbstractCamera.h"
#include "../structures/Point2.h"

namespace Polytope {
   class PerspectiveCamera : public AbstractCamera {
   public:
      PerspectiveCamera(const CameraSettings &settings, const Transform &cameraToWorld, bool leftHanded);

      Ray GetRay(Point2f pixel) override;

   public:
      float OneOverWidth;
      float OneOverHeight;
      float AspectRatio;
      float TanFOVOver2;
      bool LeftHanded = false;
   };
}

#endif //POLYTOPE_PERSPECTIVECAMERA_H
