//
// Created by Daniel Thompson on 3/8/18.
//

#include "ShapeLight.h"

#include <utility>

namespace Polytope {
   ShapeLight::ShapeLight(const class SpectralPowerDistribution &spectralPowerDistribution,
                          std::shared_ptr<AbstractShape> shape)
      : AbstractLight(spectralPowerDistribution), Shape(std::move(shape)) { }

   Point ShapeLight::GetRandomPointOnSurface() {
      return Shape->GetRandomPointOnSurface();
   }
}


