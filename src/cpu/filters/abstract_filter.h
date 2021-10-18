//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLY_ABSTRACTFILTER_H
#define POLY_ABSTRACTFILTER_H

#include <vector>
#include "../structures/Sample.h"
#include "../../common/structures/point2.h"

namespace poly {

   /**
    * The purpose of a filter is to hold all samples and then render a given pixel
    * from those samples.
    */
   class abstract_filter {
   public:
      const poly::bounds Bounds;

      explicit abstract_filter(const poly::bounds &bounds) : Bounds(bounds) { }
      virtual ~abstract_filter() = default;

      virtual void add_sample(const point2f &location, const Sample &sample) = 0;
      virtual void add_samples(const point2f &location, const std::vector<Sample> &samples) = 0;

      virtual SpectralPowerDistribution output(const point2i &pixel) const = 0;
      virtual void pre_output() { };

   };

}


#endif //POLY_ABSTRACTFILTER_H
