//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_CAMERASETTINGS_H
#define POLYTOPE_CAMERASETTINGS_H

#include "../structures/Point2.h"

namespace Polytope {

   class CameraSettings {
   public:

      // constructors
      CameraSettings(const Bounds bounds, const float fov) :
            Bounds(bounds), FieldOfView(fov) { }

      // data
      const Bounds Bounds;
      const float FieldOfView;
   };

}


#endif //POLYTOPE_CAMERASETTINGS_H
