//
// Created by daniel on 11/8/20.
//

#ifndef POLYTOPE_TEXTURE_H
#define POLYTOPE_TEXTURE_H

#include "spectrum.h"

namespace poly {

   class texture {
   public:
      
      enum format {
         RGBA,
         GREY
      };
      
      format data_format;
      
      /**
       * given (u,v), should return a color 
       */
      ReflectanceSpectrum evaluate_rgba(float u, float v) {
         const unsigned int u_texel = (unsigned int)(u * (float)width);
         const unsigned int v_texel = (unsigned int)((1.f - v) * (float)height);

         const unsigned int index = (v_texel * width + u_texel) * 4;

         return {data[index], data[index + 1], data[index + 2]};
      }

      unsigned int width, height;

      std::vector<unsigned char> data;

      /**
       * Name of this texture, as specified in the input file.
       */
      std::string name;
   };
   
}
#endif //POLYTOPE_TEXTURE_H
