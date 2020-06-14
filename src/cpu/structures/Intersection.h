//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLY_INTERSECTION_H
#define POLY_INTERSECTION_H

#include <memory>
#include "Vectors.h"

namespace poly {

   class Mesh;
   
   class Intersection {
   public:

      // methods

      Intersection();

      // data

      Mesh *Shape = nullptr;
      unsigned int faceIndex;
      Point Location;
      poly::Normal Normal;
      Vector Tangent1;
      Vector Tangent2;
      bool Hits = false;

      Vector WorldToLocal(const Vector &world) const;
      Vector LocalToWorld(const Vector &local) const;
   };

}


#endif //POLY_INTERSECTION_H
