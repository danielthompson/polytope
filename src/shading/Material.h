//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLYTOPE_MATERIAL_H
#define POLYTOPE_MATERIAL_H

#include <memory>
#include "brdf/AbstractBRDF.h"
#include "ReflectanceSpectrum.h"

namespace Polytope {

   class Material {
   public:
      Material(std::unique_ptr<AbstractBRDF> brdf, ReflectanceSpectrum reflectanceSpectrum)
            : BRDF(std::move(brdf)), ReflectanceSpectrum(reflectanceSpectrum){ }

      std::unique_ptr<AbstractBRDF> BRDF;
      ReflectanceSpectrum ReflectanceSpectrum;
   };

}

#endif //POLYTOPE_MATERIAL_H
