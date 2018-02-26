//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLYTOPE_SPECTRUM_H
#define POLYTOPE_SPECTRUM_H

namespace Polytope {

   class Spectrum {
   public:

      // constructors

      explicit Spectrum() : r(0), g(0), b(0) { };
      Spectrum(float r, float g, float b) : r(r), g(g), b(b) { };

      // operators

      Spectrum &operator+=(const Spectrum &rhs);

      // methods

      // data
      float r, g, b;

   };

}


#endif //POLYTOPE_SPECTRUM_H
