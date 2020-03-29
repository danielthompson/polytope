//
// Created by Daniel Thompson on 12/29/19.
//

#ifndef POLYTOPE_OBJEXPORTER_H
#define POLYTOPE_OBJEXPORTER_H

#include "../shapes/triangle.h"

namespace Polytope {
   class OBJExporter {
   public:
      void Export(std::ostream &stream, TriangleMesh* mesh, bool worldSpace = false);
   };
}

#endif //POLYTOPE_OBJEXPORTER_H
