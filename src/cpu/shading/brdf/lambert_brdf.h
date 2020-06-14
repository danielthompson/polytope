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
      explicit LambertBRDF() : refl(1.f, 1.f, 1.f) { };
      LambertBRDF(ReflectanceSpectrum refl) : refl(refl) { };
      
      float f(float thetaIncoming, float thetaOutgoing) const override {
//      return poly::OneOverPi;
         return 1.0f;
      }

      float f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const override {
//      return poly::OneOverPi;
         return 1.0f;
      }

      Vector sample(const Vector &incoming, ReflectanceSpectrum &refl_spectrum, float &pdf) const override {
         refl_spectrum = refl;
         return AbstractBRDF::sample(incoming, refl_spectrum, pdf);
      }

      ReflectanceSpectrum refl;
   };

}


#endif //POLY_LAMBERTBRDF_H
