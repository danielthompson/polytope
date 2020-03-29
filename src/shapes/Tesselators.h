//
// Created by Daniel Thompson on 12/28/19.
//

#ifndef POLYTOPE_TESSELATORS_H
#define POLYTOPE_TESSELATORS_H

#include "triangle.h"

namespace Polytope {
   class SphereTesselator {
   public:
      void Create(unsigned int meridians, unsigned int parallels, TriangleMesh* mesh);
   };

   class DiskTesselator {
   public:
      void Create(unsigned int meridians, TriangleMesh* mesh);
   };

   class ConeTesselator {
   public:
      void Create(unsigned int meridians, TriangleMesh* mesh);
   };
}


#endif //POLYTOPE_TESSELATORS_H
