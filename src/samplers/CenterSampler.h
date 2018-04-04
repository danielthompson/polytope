//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_CENTERSAMPLER_H
#define POLYTOPE_CENTERSAMPLER_H

#include "AbstractSampler.h"

namespace Polytope {

   class CenterSampler : public AbstractSampler {
   public:
      Point2f GetSample(int x, int y) override;

      void GetSamples(Point2f points[], int number, int x, int y) override;

      virtual ~CenterSampler() { }
   };

}

#endif //POLYTOPE_CENTERSAMPLER_H
