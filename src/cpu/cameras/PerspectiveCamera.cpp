//
// Created by dthompson on 20 Feb 18.
//

#include <cmath>
#include <iostream>
#include "PerspectiveCamera.h"
#include "../constants.h"
#include "../structures/Vectors.h"


namespace poly {

   PerspectiveCamera::PerspectiveCamera(const CameraSettings &settings, const Transform &cameraToWorld,
                                        const bool leftHanded)
         : AbstractCamera(settings, cameraToWorld),
         LeftHanded(leftHanded) {
      OneOverWidth = 1.0f / (float)Settings.Bounds.x;
      OneOverHeight = 1.0f / (float)Settings.Bounds.y;
      AspectRatio = (float) Settings.Bounds.x * OneOverHeight;
      TanFOVOver2 = (float)tan(Settings.FieldOfView * PIOver180 * .5f);
   }

   Ray PerspectiveCamera::GetRay(const Point2f pixel) {
      const float pixelNDCx = (pixel.x/* + 0.5f*/) * OneOverWidth;
      const float pixelNDCy = (pixel.y/* + 0.5f*/) * OneOverHeight;

      const float pixelCameraX = (2 * pixelNDCx - 1) * AspectRatio * TanFOVOver2;
      const float pixelCameraY = (1 - 2 * pixelNDCy) * TanFOVOver2;

      const float z = LeftHanded ? 1 : -1;

      Point imagePlanePixelInCameraSpace = { pixelCameraX, pixelCameraY, z };

      Vector direction = { imagePlanePixelInCameraSpace.x, imagePlanePixelInCameraSpace.y, imagePlanePixelInCameraSpace.z };

      direction.Normalize();

      Ray cameraSpaceRay = { DefaultOrigin, direction };
      Ray worldSpaceRay = CameraToWorld.Apply(cameraSpaceRay);

      worldSpaceRay.Direction.Normalize();
      
      // hmm... doubt
      worldSpaceRay.DirectionInverse.Normalize();

      return worldSpaceRay;
   }
}