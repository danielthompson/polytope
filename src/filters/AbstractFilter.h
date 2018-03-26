//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLYTOPE_ABSTRACTFILTER_H
#define POLYTOPE_ABSTRACTFILTER_H

#include <vector>
#include "../structures/Sample.h"
#include "../structures/Point2.h"

namespace Polytope {

   /**
    * The purpose of a filter is to hold all samples and then render a given pixel
    * from those samples.
    */
   class AbstractFilter {
   public:

      // constructors
      explicit AbstractFilter(const Polytope::Bounds &bounds) : Bounds(bounds) { }

      // methods
      virtual void AddSample(const Point2f &location, const Sample &sample) = 0;
      virtual void AddSamples(const Point2f &location, const std::vector<Sample> &samples) = 0;

      virtual SpectralPowerDistribution Output(const Point2i &pixel) = 0;

      // destructors
      virtual ~AbstractFilter() { }

      const Polytope::Bounds Bounds;
   };

}


#endif //POLYTOPE_ABSTRACTFILTER_H
