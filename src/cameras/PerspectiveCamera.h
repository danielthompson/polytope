//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_PERSPECTIVECAMERA_H
#define POLYTOPE_PERSPECTIVECAMERA_H

#include "AbstractCamera.h"

namespace Polytope {
   class PerspectiveCamera : AbstractCamera {
   public:
      PerspectiveCamera(const CameraSettings &settings, const Transform &cameraToWorld);

      Ray GetRay(float x, float y) override;

   private:
      float OneOverWidth;
      float OneOverHeight;
      float AspectRatio;
      float TanFOVOver2;
   };

}


#endif //POLYTOPE_PERSPECTIVECAMERA_H
