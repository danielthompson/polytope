//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLYTOPE_GRIDSAMPLER_H
#define POLYTOPE_GRIDSAMPLER_H

#include "AbstractSampler.h"

namespace Polytope {


   class GridSampler : public AbstractSampler {
   public:
      Point2f GetSample(int x, int y) override;

      virtual ~GridSampler() { }

      void GetSamples(Point2f *points, int number, int x, int y) override;
   };

}





#endif //POLYTOPE_GRIDSAMPLER_H
