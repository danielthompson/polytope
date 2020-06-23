//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLY_INTERSECTION_H
#define POLY_INTERSECTION_H

#include "Vectors.h"

namespace poly {

   class Mesh;
   
   class Intersection {
   public:

      Intersection() : Location(Point(0, 0, 0)) { }
      Vector WorldToLocal(const Vector &world) const;
      Vector LocalToWorld(const Vector &local) const;

      const poly::Mesh *Shape = nullptr;
      poly::Point Location;
      poly::Normal geo_normal;
      poly::Normal bent_normal;
      poly::Vector Tangent1;
      poly::Vector Tangent2;
      unsigned int faceIndex;
      bool Hits = false;
   };
}

#endif //POLY_INTERSECTION_H
