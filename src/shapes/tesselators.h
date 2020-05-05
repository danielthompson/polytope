//
// Created by Daniel Thompson on 12/28/19.
//

#ifndef POLYTOPE_TESSELATORS_H
#define POLYTOPE_TESSELATORS_H

#include "abstract_mesh.h"

namespace Polytope {
   class SphereTesselator {
   public:
      static void Create(unsigned int meridians, unsigned int parallels, AbstractMesh* mesh);
   };

   class DiskTesselator {
   public:
      static void Create(unsigned int meridians, AbstractMesh* mesh);
   };

   class ConeTesselator {
   public:
      static void Create(unsigned int meridians, AbstractMesh* mesh);
   };
}


#endif //POLYTOPE_TESSELATORS_H
