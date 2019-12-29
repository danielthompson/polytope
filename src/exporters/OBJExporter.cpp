//
// Created by Daniel Thompson on 12/29/19.
//

#include "OBJExporter.h"

namespace Polytope {

   void OBJExporter::Export(std::ostream &stream, TriangleMesh *mesh, bool worldSpace) {
      if (worldSpace) {
         for (const Point &vertex : mesh->Vertices) {
            const Point worldSpacePoint = mesh->ObjectToWorld->Apply(vertex);
            stream << "v " << worldSpacePoint.x << " " << worldSpacePoint.y << " " << worldSpacePoint.z << std::endl;
         }
      }
      else {
         for (const Point &vertex : mesh->Vertices) {
            stream << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
         }
      }

      stream << std::endl;

      for (const Point3ui &face : mesh->Faces) {
         stream << "f " << face.x << " " << face.y << " " << face.z << std::endl;
      }
   }
}
