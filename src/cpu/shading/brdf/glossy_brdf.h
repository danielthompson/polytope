//
// Created by daniel on 5/5/20.
//

#ifndef POLY_GLOSSY_BRDF_H
#define POLY_GLOSSY_BRDF_H

#include "abstract_brdf.h"
#include "../spectrum.h"

namespace poly {
   class GlossyBRDF : public AbstractBRDF {
   public:
      GlossyBRDF(const ReflectanceSpectrum &specular, const ReflectanceSpectrum &diffuse, const float roughness)
            : specular(specular), diffuse(diffuse), roughness(roughness) {  };

      float f(const float thetaIncoming, const float thetaOutgoing) const override {
         return 0;
      }

      float f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const override {
         return 0;
      }

      Vector sample(const Vector &incoming, ReflectanceSpectrum &refl_spectrum, float &pdf) const override {
         
         assert(!std::isnan(incoming.x));
         assert(!std::isnan(incoming.y));
         assert(!std::isnan(incoming.z));
         
         float lambert_pdf = 1;
         
         while (true) {
            Vector lambert_outgoing = AbstractBRDF::sample(incoming, refl_spectrum, pdf);;

            // 0. measure angle between normal and mirror
            const Normal normal = Normal(0, 1, 0);
            const float factor = incoming.Dot(normal) * 2;
            const Vector scaled = Vector(normal * factor);
            pdf = 1.0f;
            Vector specular_outgoing = incoming - scaled;
            specular_outgoing.Normalize();
            const float angle = -std::acos(specular_outgoing.Dot(normal));

            // 1. cross normal and mirror together to get a perp vector
            const Vector perp = specular_outgoing.Cross(normal);

            // 2. rotate lambert by [0] degrees around [1] axis
            Transform t = Transform::Rotate(angle, perp.x, perp.y, perp.z);
            Vector rotated_lambert = t.Apply(lambert_outgoing);

            // 3. lerp between [2] and mirror vector using roughness factor
            refl_spectrum = specular * (1 - roughness) + diffuse * roughness;
            Vector outgoing = specular_outgoing * (1 - roughness) + rotated_lambert * roughness;
            if (outgoing.y < 0)
               continue;
            //outgoing.y = outgoing.y > 0 ? outgoing.y : -outgoing.y;
            return outgoing;
         }
      }

   private:
      ReflectanceSpectrum specular, diffuse;
      
      // 0 - specular, 1 - diffuse
      float roughness;
   };
}


#endif //POLY_GLOSSY_BRDF_H
