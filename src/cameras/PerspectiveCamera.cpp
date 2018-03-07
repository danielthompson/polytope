//
// Created by dthompson on 20 Feb 18.
//

#include <cmath>
#include "PerspectiveCamera.h"
#include "../Constants.h"


namespace Polytope {

   PerspectiveCamera::PerspectiveCamera(const CameraSettings &settings, const Transform &cameraToWorld)
         : AbstractCamera(settings, cameraToWorld) {
      OneOverWidth = 1.0f / (float)Settings.x;
      OneOverHeight = 1.0f / (float)Settings.y;
      AspectRatio = (float) Settings.x * OneOverHeight;
      TanFOVOver2 = (float)tan(Settings.FieldOfView * PIOver180 * .5f);
   }

   Ray PerspectiveCamera::GetRay(const Point2f pixel) {
      float pixelNDCx = (pixel.x + 0.5f) * OneOverWidth;
      float pixelNDCy = (pixel.y + 0.5f) * OneOverHeight;

      float pixelCameraX = (2 * pixelNDCx - 1) * AspectRatio * TanFOVOver2;
      float pixelCameraY = (1 - 2 * pixelNDCy) * TanFOVOver2;

      Point imagePlanePixelInCameraSpace = Point(pixelCameraX, pixelCameraY, -1);

      Vector direction = Vector (imagePlanePixelInCameraSpace.x, imagePlanePixelInCameraSpace.y, imagePlanePixelInCameraSpace.z);

      Ray cameraSpaceRay = Ray(DefaultOrigin, direction);

      Ray worldSpaceRay = CameraToWorld.Apply(cameraSpaceRay);

      worldSpaceRay.Direction.Normalize();
      worldSpaceRay.DirectionInverse.Normalize();

      return worldSpaceRay;
   }
}