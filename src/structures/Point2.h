//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_POINT2_H
#define POLYTOPE_POINT2_H

namespace Polytope {

   template <class T>
   class Point2 {
   public:
      Point2(T x, T y) : x(x), y(y) { }
      T x, y;
   };

   typedef Point2<int> Point2i;
   typedef Point2<float> Point2f;

}

#endif //POLYTOPE_POINT2_H
