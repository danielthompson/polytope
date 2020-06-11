//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLY_CAMERASETTINGS_H
#define POLY_CAMERASETTINGS_H

#include "../../common/structures/Point2.h"

namespace poly {

   class CameraSettings {
   public:
      const poly::Bounds Bounds;
      float FieldOfView;

      CameraSettings(const poly::Bounds &bounds, const float fov) :
            Bounds(bounds), FieldOfView(fov) { }
   };
}

#endif //POLY_CAMERASETTINGS_H
