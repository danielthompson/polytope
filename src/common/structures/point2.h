//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_POINT2_H
#define POLY_POINT2_H

namespace poly {

   template <class T>
   class point2 {
   public:
      point2() : x(0), y(0) { }
      point2(T x, T y) : x(x), y(y) { }
      T x, y;
   };

   typedef point2<int> point2i;
   typedef point2<float> point2f;
   typedef point2<unsigned int> bounds;

}

#endif //POLY_POINT2_H
