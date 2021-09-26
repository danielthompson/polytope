//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLY_CAMERASETTINGS_H
#define POLY_CAMERASETTINGS_H

#include "../../common/structures/point2.h"

namespace poly {

   class camera_settings {
   public:
      const poly::bounds bounds;
      float field_of_view;

      camera_settings(const poly::bounds &bounds, const float fov) :
            bounds(bounds), field_of_view(fov) { }
   };
}

#endif //POLY_CAMERASETTINGS_H
