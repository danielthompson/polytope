//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_ABSTRACTCAMERA_H
#define POLYTOPE_ABSTRACTCAMERA_H

#include "../structures/Vector.h"
#include "../structures/Point2.h"
#include "../structures/Transform.h"
#include "CameraSettings.h"

namespace Polytope {

   class AbstractCamera {
   public:

      // methods
      AbstractCamera(const CameraSettings &settings, const Transform &cameraToWorld);

      virtual Ray GetRay(Point2f pixel) = 0;

      // data
      const CameraSettings Settings;
      const Transform CameraToWorld;


   protected:
      const Vector DefaultDirection = Vector(0, 0, -1);
      const Point DefaultOrigin = Point(0, 0, 0);
   };
}


#endif //POLYTOPE_ABSTRACTCAMERA_H
