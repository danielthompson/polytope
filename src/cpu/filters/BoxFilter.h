//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLY_BOXFILTER_H
#define POLY_BOXFILTER_H

#include <vector>
#include <mutex>
#include "AbstractFilter.h"
#include "../../common/structures/Point2.h"

namespace poly {

   class BoxFilter : public AbstractFilter {
   public:

      explicit BoxFilter(const poly::Bounds &bounds)
            : AbstractFilter(bounds), _data(bounds.x * bounds.y, std::vector<Sample>()) { }

      void SetSamples(unsigned int samples);
      SpectralPowerDistribution Output(const Point2i &pixel) const override;
      void AddSample(const Point2f &location, const Sample &sample) override;
      void AddSamples(const Point2f &location, const std::vector<Sample> &samples) override;

   private:
      std::vector<std::vector<Sample>> _data;
   };
}

#endif //POLY_BOXFILTER_H
