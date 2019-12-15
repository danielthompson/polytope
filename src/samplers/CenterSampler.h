//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_CENTERSAMPLER_H
#define POLYTOPE_CENTERSAMPLER_H

#include "AbstractSampler.h"

namespace Polytope {

   class CenterSampler : public AbstractSampler {
   public:
      Point2f GetSample(int x, int y) const override;

      void GetSamples(Point2f points[], unsigned int number, int x, int y) const override;

      ~CenterSampler() override = default;
   };

}

#endif //POLYTOPE_CENTERSAMPLER_H
