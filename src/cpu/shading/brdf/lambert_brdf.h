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
            refl_spectrum = texture->evaluate_rgba(u, v);
         }
         return AbstractBRDF::sample(incoming, u, v, refl_spectrum, pdf);
      }

      ReflectanceSpectrum refl;
   };
}

#endif //POLY_LAMBERTBRDF_H
