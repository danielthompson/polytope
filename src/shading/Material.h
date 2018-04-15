//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLYTOPE_MATERIAL_H
#define POLYTOPE_MATERIAL_H

#include <memory>
#include <utility>
#include "brdf/AbstractBRDF.h"
#include "ReflectanceSpectrum.h"

namespace Polytope {

   class Material {
   public:
      explicit Material() = default;

      explicit Material(std::string name) : Name(std::move(name)) { }

      Material(std::shared_ptr<AbstractBRDF> brdf, Polytope::ReflectanceSpectrum reflectanceSpectrum)
            : BRDF(std::move(brdf)), ReflectanceSpectrum(reflectanceSpectrum){ }

      std::shared_ptr<AbstractBRDF> BRDF;
      Polytope::ReflectanceSpectrum ReflectanceSpectrum;
      std::string Name;
   };

}

#endif //POLYTOPE_MATERIAL_H
