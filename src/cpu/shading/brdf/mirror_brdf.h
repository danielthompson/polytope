//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLY_MIRRORBRDF_H
#define POLY_MIRRORBRDF_H

#include "abstract_brdf.h"
#include "../texture.h"

namespace poly {

   class MirrorBRDF : public AbstractBRDF {
   public:
      std::shared_ptr<poly::texture> texture;
      MirrorBRDF(const ReflectanceSpectrum &refl) : AbstractBRDF(BRDF_TYPE::Mirror), refl(refl) { }
      float f(const float thetaIncoming, const float thetaOutgoing) const override {
         if (poly::WithinEpsilon(thetaIncoming, thetaOutgoing))
            return 1.0f;
         return 0.0f;
      }

      Vector sample(const Vector &incoming, const float u, const float v, poly::ReflectanceSpectrum &refl_spectrum, float &pdf) const override {
         //const Normal normal = Normal(0, 1, 0);
//         const float factor = incoming.Dot(normal) * 2;
//         const Vector scaled = Vector(normal * factor);
//         pdf = 1.0f;
//         const Vector outgoing = incoming - scaled;
//         return outgoing;

         refl_spectrum = refl;

         pdf = 1.0f;
         return {incoming.x, -incoming.y, incoming.z};
      }

      float f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const override {

         const float thetaIncoming = RadiansBetween(-incoming, normal);
         const float thetaOutgoing = RadiansBetween(outgoing, normal);

         return f(thetaIncoming, thetaOutgoing);
      }
      
      ReflectanceSpectrum refl;
   private:
      const Normal normal = Normal(0, 1, 0);
   };

}

#endif //POLY_MIRRORBRDF_H
