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
            : AbstractBRDF(BRDF_TYPE::Glossy), specular(specular), diffuse(diffuse), roughness(roughness) {  };

      float f(const float thetaIncoming, const float thetaOutgoing) const override {
         return 0;
      }

      float f(const poly::vector &incoming, const poly::normal &normal, const vector &outgoing) const override {
         return 0;
      }

      poly::vector sample(const poly::vector &incoming, const float u, const float v, poly::ReflectanceSpectrum &refl_spectrum, float &pdf) const override {
         
         assert(!std::isnan(incoming.x));
         assert(!std::isnan(incoming.y));
         assert(!std::isnan(incoming.z));
         
         float lambert_pdf = 1;
         
         while (true) {
            poly::vector lambert_outgoing = AbstractBRDF::sample(incoming, u, v, refl_spectrum, pdf);;

            // 0. measure angle between normal and mirror
            const poly::normal normal(0, 1, 0);
            const float factor = incoming.dot(normal) * 2;
            const poly::vector scaled = vector(normal * factor);
            pdf = 1.0f;
            poly::vector specular_outgoing = incoming - scaled;
            specular_outgoing.normalize();
            const float angle = -std::acos(specular_outgoing.dot(normal));

            // 1. cross normal and mirror together to get a perp vector
            const poly::vector perp = specular_outgoing.cross(normal);

            // 2. rotate lambert by [0] degrees around [1] axis
            poly::transform t = poly::transform::rotate(angle, perp.x, perp.y, perp.z);
            poly::vector rotated_lambert = t.apply(lambert_outgoing);

            // 3. lerp between [2] and mirror vector using roughness factor
            refl_spectrum = specular * (1 - roughness) + diffuse * roughness;
            poly::vector outgoing = specular_outgoing * (1 - roughness) + rotated_lambert * roughness;
            if (outgoing.y < 0)
               continue;
            //outgoing.y = outgoing.y > 0 ? outgoing.y : -outgoing.y;
            return outgoing;
         }
      }

   private:
      poly::ReflectanceSpectrum specular, diffuse;
      
      // 0 - specular, 1 - diffuse
      float roughness;
   };
}


#endif //POLY_GLOSSY_BRDF_H
