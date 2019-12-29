//
// Created by Daniel Thompson on 12/28/19.
//

#include "SphereTesselator.h"

namespace Polytope {

   void SphereTesselator::Create(unsigned int meridians, unsigned int parallels, TriangleMesh *mesh) {

      std::vector<unsigned int> parallelStartIndices;

      // north pole
      mesh->Vertices.emplace_back(0, 1, 0);
      parallelStartIndices.push_back(0);

      // TODO make these in radians to begin with
      const float meridianAngleStep = 360 / (float)meridians;
      const float parallelAngleStep = 180 / (float)(parallels + 1);

      for (unsigned int i = 0; i < parallels; i++) {
         const float verticalAngleInDegrees = 90 - parallelAngleStep * ((float)(i + 1));
         const float verticalAngleInRadians = verticalAngleInDegrees * PIOver180;
         const float y = std::sin(verticalAngleInRadians);
         const float xzRadiusAtY = std::cos(verticalAngleInRadians);
         const unsigned int parallelStartIndex = i * meridians + 1;
         parallelStartIndices.push_back(parallelStartIndex);
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

      const int numVertices = mesh->Vertices.size();

      // faces

      // top band
      for (unsigned int v = 1; v < meridians; v++) {
         // obj face indexes start at 1
         mesh->Faces.emplace_back(v + 2, v + 1, 1);
      }
      // last face in the top band
      mesh->Faces.emplace_back(2, meridians + 1, 1);

      // middle bands
      for (unsigned int p = 1; p < parallels; p++) {
         const unsigned int topStartIndex = parallelStartIndices[p];
         const unsigned int bottomStartIndex = parallelStartIndices[p + 1];
         for (unsigned int m = 1; m < meridians; m++) {
            // obj face indexes start at 1
            const unsigned int topRightIndex = topStartIndex + m ;
            const unsigned int topLeftIndex = topStartIndex + m + 1;
            const unsigned int bottomLeftIndex = bottomStartIndex + m;
            const unsigned int bottomRightIndex = bottomStartIndex + m + 1;
            mesh->Faces.emplace_back(bottomRightIndex, bottomLeftIndex, topRightIndex);
            mesh->Faces.emplace_back(topRightIndex, topLeftIndex, bottomRightIndex);
         }
         // final meridian
         const unsigned int topLeftIndex = topStartIndex + 1;
         const unsigned int topRightIndex = topStartIndex  + meridians;
         const unsigned int bottomLeftIndex = bottomStartIndex + 1;
         const unsigned int bottomRightIndex = bottomStartIndex  + meridians;
         mesh->Faces.emplace_back(bottomLeftIndex, bottomRightIndex, topRightIndex);
         mesh->Faces.emplace_back(topRightIndex, topLeftIndex, bottomLeftIndex);
      }

      const int bottomStartIndex = parallelStartIndices[parallels];

      // bottom band
      for (unsigned int v = 1; v < meridians; v++) {
         // obj face indexes start at 1
         mesh->Faces.emplace_back(v + bottomStartIndex, v + 1 + bottomStartIndex, numVertices);
      }
      // last face in the bottom band
      mesh->Faces.emplace_back(numVertices - 1, bottomStartIndex + 1, numVertices);
   }
}