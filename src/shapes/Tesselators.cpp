//
// Created by Daniel Thompson on 12/28/19.
//

#include "Tesselators.h"

namespace Polytope {

   void SphereTesselator::Create(unsigned int meridians, unsigned int parallels, TriangleMesh *mesh) {

      std::vector<unsigned int> parallelStartIndices;

      // north pole
      mesh->Vertices.emplace_back(0, 1, 0);
      parallelStartIndices.push_back(0);

      const float meridianAngleStep = TwoPI / (float)meridians;
      const float parallelAngleStep = PI / (float)(parallels + 1);

      for (unsigned int i = 0; i < parallels; i++) {
         const float verticalAngleInRadians = PIOver2 - (parallelAngleStep * ((float)(i + 1)));
         const float y = std::sin(verticalAngleInRadians);
         const float xzRadiusAtY = std::cos(verticalAngleInRadians);
         const unsigned int parallelStartIndex = i * meridians + 1;
         parallelStartIndices.push_back(parallelStartIndex);
         for (unsigned int j = 0; j < meridians; j++) {
            const float horizontalAngle = meridianAngleStep * j;
            const float x = std::cos(horizontalAngle) * xzRadiusAtY;
            const float z = std::sin(horizontalAngle) * xzRadiusAtY;
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
         mesh->Faces.emplace_back(v + 1, v, 0);
      }
      // last face in the top band
      mesh->Faces.emplace_back(1, meridians, 0);

      // middle bands
      for (unsigned int p = 1; p < parallels; p++) {
         const unsigned int topStartIndex = parallelStartIndices[p];
         const unsigned int bottomStartIndex = parallelStartIndices[p + 1];
         for (unsigned int m = 1; m < meridians; m++) {
            // obj face indexes start at 1
            const unsigned int topRightIndex = topStartIndex + m - 1;
            const unsigned int topLeftIndex = topStartIndex + m;
            const unsigned int bottomLeftIndex = bottomStartIndex + m - 1;
            const unsigned int bottomRightIndex = bottomStartIndex + m;
            mesh->Faces.emplace_back(bottomRightIndex, bottomLeftIndex, topRightIndex);
            mesh->Faces.emplace_back(topRightIndex, topLeftIndex, bottomRightIndex);
         }
         // final meridian
         const unsigned int topLeftIndex = topStartIndex;
         const unsigned int topRightIndex = topStartIndex  + meridians - 1;
         const unsigned int bottomLeftIndex = bottomStartIndex;
         const unsigned int bottomRightIndex = bottomStartIndex  + meridians - 1;
         mesh->Faces.emplace_back(bottomLeftIndex, bottomRightIndex, topRightIndex);
         mesh->Faces.emplace_back(topRightIndex, topLeftIndex, bottomLeftIndex);
      }

      const int bottomStartIndex = parallelStartIndices[parallels];

      // bottom band
      for (unsigned int v = 0; v < meridians - 1; v++) {
         // obj face indexes start at 1
         mesh->Faces.emplace_back(v + bottomStartIndex, v + 1 + bottomStartIndex, numVertices - 1);
      }
      // last face in the bottom band
      mesh->Faces.emplace_back(numVertices - 2, bottomStartIndex, numVertices - 1);
   }

   void DiskTesselator::Create(unsigned int meridians, TriangleMesh *mesh) {
      // center
      mesh->Vertices.emplace_back(0, 0, 0);

      const float meridianAngleStep = -TwoPI / (float)meridians;

      for (unsigned int m = 0; m < meridians; m++) {
         const float angle = meridianAngleStep * m;
         const float x = std::cos(angle);
         const float z = std::sin(angle);
         mesh->Vertices.emplace_back(x, 0, z);
      }

      for (unsigned int m = 1; m < meridians; m++) {
         mesh->Faces.emplace_back(0, m, m + 1);
      }
      mesh->Faces.emplace_back(0, meridians, 1);
   }

   void ConeTesselator::Create(unsigned int meridians, TriangleMesh *mesh) {
      // center
      mesh->Vertices.emplace_back(0, 1, 0);

      const float meridianAngleStep = -TwoPI / (float)meridians;

      for (unsigned int m = 0; m < meridians; m++) {
         const float angle = meridianAngleStep * m;
         const float x = std::cos(angle);
         const float z = std::sin(angle);
         mesh->Vertices.emplace_back(x, 0, z);
      }

      for (unsigned int m = 1; m < meridians; m++) {
         mesh->Faces.emplace_back(0, m, m + 1);
      }
      mesh->Faces.emplace_back(0, meridians, 1);
   }
}