//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractCamera.h"

namespace Polytope {

   AbstractCamera::AbstractCamera(const CameraSettings &settings, const Transform &cameraToWorld)
         : Settings(settings), CameraToWorld(cameraToWorld) { }

}
