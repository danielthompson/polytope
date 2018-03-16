//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLYTOPE_ABSTRACTSAMPLER_H
#define POLYTOPE_ABSTRACTSAMPLER_H

#include "../structures/Point2.h"

namespace Polytope {

   class AbstractSampler {
   public:

      // methods
      virtual Point2f GetSample(int x, int y) = 0;

      virtual void GetSamples(Point2f *points, int number, int x, int y) = 0;

      // destructors

      virtual ~AbstractSampler() { }

   };

}


#endif //POLYTOPE_ABSTRACTSAMPLER_H
