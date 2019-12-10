//
// Created by Daniel on 21-Apr-18.
//

#ifndef POLYTOPE_SHAPELIGHT_H
#define POLYTOPE_SHAPELIGHT_H

#include "AbstractLight.h"

namespace Polytope {

   class ShapeLight : public AbstractLight {
   public:
      explicit ShapeLight(const Polytope::SpectralPowerDistribution &spectralPowerDistribution)
            : AbstractLight(spectralPowerDistribution) {}

   };

}


#endif //POLYTOPE_SHAPELIGHT_H
