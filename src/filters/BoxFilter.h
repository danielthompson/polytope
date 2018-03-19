//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLYTOPE_BOXFILTER_H
#define POLYTOPE_BOXFILTER_H

#include <vector>
#include "boost/multi_array.hpp"
#include "AbstractFilter.h"
#include "../structures/Point2.h"


namespace Polytope {

   class BoxFilter : public AbstractFilter {
   public:
      explicit BoxFilter(const Polytope::Bounds &bounds)
            : AbstractFilter(bounds), _data(bounds.x * bounds.y, std::vector<Sample>(0)){ }

      SpectralPowerDistribution Output(const Point2i &pixel) override;

      void AddSample(const Point2f &location, const Sample &sample) override;

      void AddSamples(const Point2f &location, const std::vector<Sample> &samples) override;

   private:
      std::vector<std::vector<Sample>> _data;
   };

}


#endif //POLYTOPE_BOXFILTER_H
