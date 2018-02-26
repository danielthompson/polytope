//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_NORMAL_H
#define POLYTOPE_NORMAL_H

namespace Polytope {

   class Normal {
   public:

      // constructors

      Normal () {x = y = z = 0;}

      Normal (float x, float y, float z) : x(x), y(y), z(z) { }

      // operators

      Normal operator*(const float t);

      // methods

      float Length();
      float LengthSquared();

      void Normalize();

      // data

      float x, y, z;
   };

}

#endif //POLYTOPE_NORMAL_H
