//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_POINT2_H
#define POLY_POINT2_H

namespace poly {

   template <class T>
   class Point2 {
   public:
      Point2() : x(0), y(0) { }
      Point2(T x, T y) : x(x), y(y) { }
      T x, y;
   };

   typedef Point2<int> Point2i;
   typedef Point2<float> Point2f;
   typedef Point2<unsigned int> Bounds;

}

#endif //POLY_POINT2_H
