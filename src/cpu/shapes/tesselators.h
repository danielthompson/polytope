//
// Created by Daniel Thompson on 12/28/19.
//

#ifndef POLY_TESSELATORS_H
#define POLY_TESSELATORS_H

#include "mesh.h"

namespace poly {
   class SphereTesselator {
   public:
      static void Create(unsigned int meridians, unsigned int parallels, std::shared_ptr<poly::mesh_geometry> mesh);
   };

   class DiskTesselator {
   public:
      static void Create(unsigned int meridians, std::shared_ptr<poly::mesh_geometry> mesh);
   };

   class ConeTesselator {
   public:
      static void Create(unsigned int meridians, std::shared_ptr<poly::mesh_geometry> mesh);
   };
}


#endif //POLY_TESSELATORS_H
