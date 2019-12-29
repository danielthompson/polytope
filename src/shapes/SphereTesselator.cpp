//
// Created by Daniel Thompson on 12/28/19.
//

#include "SphereTesselator.h"

namespace Polytope {

   void SphereTesselator::Create(unsigned int meridians, unsigned int parallels, TriangleMesh *mesh) {
      // north pole
      mesh->Vertices.emplace_back(0, 1, 0);

      // TODO make these in radians to begin with
      const float meridianAngleStep = 360 / (float)meridians;
      const float parallelAngleStep = 180 / (float)(parallels + 1);

      for (unsigned int i = 0; i < parallels; i++) {
         const float verticalAngleInDegrees = 90 - parallelAngleStep * ((float)(i + 1));
         const float verticalAngleInRadians = verticalAngleInDegrees * PIOver180;
         const float y = std::sin(verticalAngleInRadians);
         const float xzRadiusAtY = std::cos(verticalAngleInRadians);
         for (unsigned int j = 0; j < meridians; j++) {
            const float horizontalAngleInDegrees = meridianAngleStep * j;
            const float horizontalAngleInRadians = horizontalAngleInDegrees * PIOver180;
            const float x = std::cos(horizontalAngleInRadians) * xzRadiusAtY;
            const float z = std::sin(horizontalAngleInRadians) * xzRadiusAtY;
            mesh->Vertices.emplace_back(x, y, z);
         }
      }

      // south pole
      mesh->Vertices.emplace_back(0, -1, 0);
   }
}