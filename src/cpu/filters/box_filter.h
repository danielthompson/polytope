//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLY_BOXFILTER_H
#define POLY_BOXFILTER_H

#include <vector>
#include <mutex>
#include "abstract_filter.h"
#include "../../common/structures/point2.h"

namespace poly {

   class box_filter : public abstract_filter {
   public:

      explicit box_filter(const poly::bounds &bounds)
            : abstract_filter(bounds), _data(bounds.x * bounds.y, std::vector<Sample>()) { }

      SpectralPowerDistribution output(const point2i &pixel) const override;
      void add_sample(const point2f &location, const Sample &sample) override;
      void add_samples(const point2f &location, const std::vector<Sample> &samples) override;

   private:
      std::vector<std::vector<Sample>> _data;
   };
}

#endif //POLY_BOXFILTER_H
