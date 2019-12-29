//
// Created by Daniel Thompson on 12/28/19.
//

#ifndef POLYTOPE_SPHERETESSELATOR_H
#define POLYTOPE_SPHERETESSELATOR_H

#include "TriangleMesh.h"

namespace Polytope {
   class SphereTesselator {
   public:
      void Create(unsigned int meridians, unsigned int parallels, TriangleMesh* mesh);
   };
}


#endif //POLYTOPE_SPHERETESSELATOR_H
