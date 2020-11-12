//
// Created by Daniel on 31-Mar-18.
//

#ifndef POLY_LAMBERTBRDF_H
#define POLY_LAMBERTBRDF_H

#include "abstract_brdf.h"
#include "../../constants.h"

namespace poly {

   class LambertBRDF : public AbstractBRDF {
   public:
      std::shared_ptr<poly::texture> texture;
      
      explicit LambertBRDF() : 
         AbstractBRDF(BRDF_TYPE::Lambert),
         refl(1.f, 1.f, 1.f),
         texture(nullptr) {
         // intentionally empty
      };
      
      LambertBRDF(ReflectanceSpectrum refl) : 
         AbstractBRDF(BRDF_TYPE::Lambert), 
         refl(refl),
         texture(nullptr) {
         // intentionally empty
      };
      
      LambertBRDF(std::shared_ptr<poly::texture> texture) :
         AbstractBRDF(BRDF_TYPE::Lambert),
         texture(texture) {
         // intentionally empty
      };
         
      
      float f(float thetaIncoming, float thetaOutgoing) const override {
//      return poly::OneOverPi;
         return 1.0f;
      }

      float f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const override {
//      return poly::OneOverPi;
         return 1.0f;
      }

      Vector sample(const Vector &incoming, const float u, const float v, ReflectanceSpectrum &refl_spectrum, float &pdf) const override {
         
         if (texture == nullptr) {
            refl_spectrum = refl;   
         }
         else {
            const unsigned int u_texel = (unsigned int)(u * (float)texture->width);
            const unsigned int v_texel = (unsigned int)((1.f - v) * (float)texture->height);
            
            const unsigned int index = (v_texel * texture->width + u_texel) * 4;
            
            refl_spectrum = { texture->data[index], texture->data[index + 1],texture->data[index + 2]};
         }
         return AbstractBRDF::sample(incoming, u, v, refl_spectrum, pdf);
      }

      ReflectanceSpectrum refl;
   };
}

#endif //POLY_LAMBERTBRDF_H
