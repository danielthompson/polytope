//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLYTOPE_GRIDSAMPLER_H
#define POLYTOPE_GRIDSAMPLER_H

#include "AbstractSampler.h"

namespace Polytope {


   class GridSampler : public AbstractSampler {
   public:
      Point2f GetSample(int x, int y) const override;

      ~GridSampler() override = default;

      void GetSamples(Point2f points[], unsigned int number, int x, int y) const override;
   };

}





#endif //POLYTOPE_GRIDSAMPLER_H
