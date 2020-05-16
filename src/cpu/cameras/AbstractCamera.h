//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_ABSTRACTCAMERA_H
#define POLYTOPE_ABSTRACTCAMERA_H

#include "../structures/Vectors.h"
#include "../../common/structures/Point2.h"
#include "../structures/Transform.h"
#include "CameraSettings.h"

namespace Polytope {

   class AbstractCamera {
   public:

      AbstractCamera(const CameraSettings &settings, const Transform &cameraToWorld)
            : Settings(settings), CameraToWorld(cameraToWorld) { }
      virtual ~AbstractCamera() = default;

      virtual Ray GetRay(Point2f pixel) = 0;

      const CameraSettings Settings;
      const Transform CameraToWorld;

      Point eye;
      Point lookAt;
      Vector up;

   protected:
      const Vector DefaultDirection = Vector(0, 0, -1);
      const Point DefaultOrigin = Point(0, 0, 0);
   };
}


#endif //POLYTOPE_ABSTRACTCAMERA_H
