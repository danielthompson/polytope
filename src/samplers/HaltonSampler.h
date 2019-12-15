//
// Created by Daniel on 04-Apr-18.
//

#ifndef POLYTOPE_HALTONSAMPLER_H
#define POLYTOPE_HALTONSAMPLER_H

#include "AbstractSampler.h"

namespace Polytope {

   class HaltonSampler : public AbstractSampler {
   public:
      Point2f GetSample(int x, int y) const override;

      void GetSamples(Point2f points[], unsigned int number, int x, int y) const override;

   };

}


#endif //POLYTOPE_HALTONSAMPLER_H
