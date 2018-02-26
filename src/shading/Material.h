//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLYTOPE_MATERIAL_H
#define POLYTOPE_MATERIAL_H

#include "brdf/AbstractBRDF.h"

namespace Polytope {

   class Material {
   public:
      AbstractBRDF BRDF;
   };

}

#endif //POLYTOPE_MATERIAL_H
