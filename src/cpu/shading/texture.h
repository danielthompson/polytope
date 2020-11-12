//
// Created by daniel on 11/8/20.
//

#ifndef POLYTOPE_TEXTURE_H
#define POLYTOPE_TEXTURE_H

#include "spectrum.h"

namespace poly {

   class texture {
   public:
      /**
       * given (u,v), should return a color 
       */
      ReflectanceSpectrum evaluate(float u, float v) {
         // TODO
         return ReflectanceSpectrum();
      }

      unsigned int width, height;

      /**
       * 4 bytes per pixel, RGBA.
       */
      std::vector<unsigned char> data;

      /**
       * Name of this texture, as specified in the input file.
       */
      std::string name;
   };
   
}
#endif //POLYTOPE_TEXTURE_H
