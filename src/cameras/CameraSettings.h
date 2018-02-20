//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_CAMERASETTINGS_H
#define POLYTOPE_CAMERASETTINGS_H

namespace Polytope {

   class CameraSettings {
   public:

      // methods
      CameraSettings(int x, int y, float fov);

      // data
      const int x;
      const int y;
      const float FieldOfView;


   };

}


#endif //POLYTOPE_CAMERASETTINGS_H
