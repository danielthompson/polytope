//
// Created by Daniel Thompson on 12/28/19.
//

#include "tesselators.h"
#include "mesh.h"

namespace poly {

   void SphereTesselator::Create(unsigned int meridians, unsigned int parallels, poly::Mesh *mesh) {

      std::vector<unsigned int> parallelStartIndices;

      // north pole
      mesh->add_vertex(0, 1, 0);
      mesh->add_vertex(0, 1, 0);
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
            mesh->add_vertex(x, y, z);
         }
      }

      // south pole
      mesh->add_vertex(0, -1, 0);

      //const int numVertices = mesh->Vertices.size();

      // faces

      // top band
      for (unsigned int v = 1; v < meridians; v++) {
         // obj face indexes start at 1
         mesh->add_packed_face(v + 1, v, 0);
      }
      // last face in the top band
      mesh->add_packed_face(1, meridians, 0);

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
            mesh->add_packed_face(bottomRightIndex, bottomLeftIndex, topRightIndex);
            mesh->add_packed_face(topRightIndex, topLeftIndex, bottomRightIndex);
         }
         // final meridian
         const unsigned int topLeftIndex = topStartIndex;
         const unsigned int topRightIndex = topStartIndex  + meridians - 1;
         const unsigned int bottomLeftIndex = bottomStartIndex;
         const unsigned int bottomRightIndex = bottomStartIndex  + meridians - 1;
         mesh->add_packed_face(bottomLeftIndex, bottomRightIndex, topRightIndex);
         mesh->add_packed_face(topRightIndex, topLeftIndex, bottomLeftIndex);
      }

      const int bottomStartIndex = parallelStartIndices[parallels];

      // bottom band
      for (unsigned int v = 0; v < meridians - 1; v++) {
         // obj face indexes start at 1
         mesh->add_packed_face(v + bottomStartIndex, v + 1 + bottomStartIndex, mesh->num_vertices_packed - 1);
      }
      // last face in the bottom band
      mesh->add_packed_face(mesh->num_vertices_packed - 2, bottomStartIndex, mesh->num_vertices_packed - 1);
      
      for (int i = 0; i < mesh->num_vertices_packed; i++) {
         Point p = mesh->get_vertex(i);
         mesh->ObjectToWorld->ApplyInPlace(p);
         
      }
      
      mesh->unpack_faces();
   }

   void DiskTesselator::Create(unsigned int meridians, poly::Mesh *mesh) {
      // center
      mesh->add_vertex(0, 0, 0);

      const float meridianAngleStep = -TwoPI / (float)meridians;

      for (unsigned int m = 0; m < meridians; m++) {
         const float angle = meridianAngleStep * m;
         const float x = std::cos(angle);
         const float z = std::sin(angle);
         mesh->add_vertex(x, 0, z);
      }

      for (unsigned int m = 1; m < meridians; m++) {
         mesh->add_packed_face(0, m, m + 1);
      }
      mesh->add_packed_face(0, meridians, 1);
   }

   void ConeTesselator::Create(unsigned int meridians, poly::Mesh *mesh) {
      // center
      mesh->add_vertex(0, 1, 0);

      const float meridianAngleStep = -TwoPI / (float)meridians;

      for (unsigned int m = 0; m < meridians; m++) {
         const float angle = meridianAngleStep * m;
         const float x = std::cos(angle);
         const float z = std::sin(angle);
         mesh->add_vertex(x, 0, z);
      }

      for (unsigned int m = 1; m < meridians; m++) {
         mesh->add_packed_face(0, m, m + 1);
      }
      mesh->add_packed_face(0, meridians, 1);
   }
}