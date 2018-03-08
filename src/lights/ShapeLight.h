//
// Created by Daniel Thompson on 3/8/18.
//

#ifndef POLYTOPE_SHAPELIGHT_H
#define POLYTOPE_SHAPELIGHT_H

#include "AbstractLight.h"
#include "../shapes/AbstractShape.h"

namespace Polytope {

   class ShapeLight : public AbstractLight {
   public:
      explicit ShapeLight(const class SpectralPowerDistribution &spectralPowerDistribution,
                          std::shared_ptr<AbstractShape> shape);

      Point GetRandomPointOnSurface() override;

      std::shared_ptr<AbstractShape> Shape;

   };

}


#endif //POLYTOPE_SHAPELIGHT_H
