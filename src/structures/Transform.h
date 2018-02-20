//
// Created by Daniel Thompson on 2/19/18.
//

#ifndef POLYTOPE_TRANSFORM_H
#define POLYTOPE_TRANSFORM_H

#include "Matrix4x4.h"

namespace Polytope::Structures {

   class Transform {
   public:
      bool operator==(const Transform &rhs) const;

      bool operator!=(const Transform &rhs) const;

      Matrix4x4 Matrix;
      Matrix4x4 Inverse;

      Transform();
      explicit Transform(const float values[4][4]);
      explicit Transform(const Matrix4x4 matrix);
      explicit Transform(const Matrix4x4 matrix, const Matrix4x4 inverse);

      Transform Invert();
   };

}

#endif //POLYTOPE_TRANSFORM_H
