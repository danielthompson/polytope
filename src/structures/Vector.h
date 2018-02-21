//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_VECTOR_H
#define POLYTOPE_VECTOR_H

namespace Polytope {

   class Vector {
   public:
      Vector(float x, float y, float z);

      Vector(const Vector &v);

      bool operator==(const Vector &rhs) const;
      bool operator!=(const Vector &rhs) const;
      float operator[] (int index) const;
      Vector operator*(const float t) const;


      float Dot(const Vector &v) const;

      float Length() const;
      float LengthSquared() const;

   public:
      float x, y, z;
   };

}

#endif //POLYTOPE_VECTOR_H
