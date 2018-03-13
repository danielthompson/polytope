//
// Created by Daniel Thompson on 2/28/18.
//

#ifndef POLYTOPE_REFLECTANCESPECTRUM_H
#define POLYTOPE_REFLECTANCESPECTRUM_H

namespace Polytope {

   class ReflectanceSpectrum {
   public:

      // constructors
      explicit ReflectanceSpectrum() : r(0), g(0), b(0) { };
      ReflectanceSpectrum(float r, float g, float b) : r(r), g(g), b(b) { };

      // operators
      ReflectanceSpectrum &operator+=(const ReflectanceSpectrum &rhs);
      ReflectanceSpectrum operator+(const ReflectanceSpectrum &rhs);
      ReflectanceSpectrum &operator*=(const ReflectanceSpectrum &rhs);
      ReflectanceSpectrum operator*(const float t);
      ReflectanceSpectrum operator*=(const float t);
      ReflectanceSpectrum operator*(const ReflectanceSpectrum &rhs);

      // methods
      ReflectanceSpectrum Lerp(const float w1, const ReflectanceSpectrum &s2, const float w2);

      // data
      float r, g, b;
   };

}


#endif //POLYTOPE_REFLECTANCESPECTRUM_H
