//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLY_MATERIAL_H
#define POLY_MATERIAL_H

#include <memory>
#include <utility>
#include "brdf/abstract_brdf.h"
#include "spectrum.h"
#include "texture.h"

namespace poly {
   
   class Material {
   public:
      explicit Material() = default;

      explicit Material(std::string name) : Name(std::move(name)) { }

      Material(std::shared_ptr<poly::AbstractBRDF> brdf)
            : BRDF(std::move(brdf)){ }

      Material(
            std::shared_ptr<poly::AbstractBRDF> brdf,
            poly::ReflectanceSpectrum reflectanceSpectrum,
            std::string name)
            : BRDF(std::move(brdf)),
              Name (std::move(name)){ }

      std::shared_ptr<poly::AbstractBRDF> BRDF;
      std::string Name;
      
   };
}

#endif //POLY_MATERIAL_H
