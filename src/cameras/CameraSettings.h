//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_CAMERASETTINGS_H
#define POLYTOPE_CAMERASETTINGS_H

#include "../structures/Point2.h"

namespace Polytope {

   class CameraSettings {
   public:
      const Polytope::Bounds Bounds;
      float FieldOfView;

      CameraSettings(const Polytope::Bounds &bounds, const float fov) :
            Bounds(bounds), FieldOfView(fov) { }
   };
}

#endif //POLYTOPE_CAMERASETTINGS_H
