//
// Created by Daniel Thompson on 2/24/18.
//

#include "Spectrum.h"

namespace Polytope {

   Spectrum &Spectrum::operator+=(const Spectrum &rhs) {
      r += rhs.r;
      g += rhs.g;
      b += rhs.b;
      return *this;
   }

}