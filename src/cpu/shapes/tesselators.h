//
// Created by Daniel Thompson on 12/28/19.
//

#ifndef POLY_TESSELATORS_H
#define POLY_TESSELATORS_H

#include "abstract_mesh.h"

namespace poly {
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


#endif //POLY_TESSELATORS_H
