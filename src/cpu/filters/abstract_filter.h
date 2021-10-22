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

      explicit abstract_filter(
            const poly::bounds &bounds, 
            const float x_width,
            const float y_width) 
            : Bounds(bounds),
              x_width(x_width),
              y_width(y_width),
              _output(std::vector<std::pair<poly::SpectralPowerDistribution, float>>(bounds.x * bounds.y))
              { }

      virtual ~abstract_filter() = default;

      virtual void add_sample(const point2i& pixel, const point2f &location, const Sample &sample) {
         const float weight = evaluate(pixel, location);
         const int index = pixel.y * Bounds.x + pixel.x;
         _output[index].first += (sample.SpectralPowerDistribution * weight);
         _output[index].second += weight;
      };
      
      virtual SpectralPowerDistribution output(const point2i &pixel) const {
         const unsigned int index = pixel.y * Bounds.x + pixel.x;
         return _output[index].first * (1.f / _output[index].second);
      };

      virtual void pre_output() { };
      
   protected:
      virtual float evaluate(const point2i& pixel, const point2f& location) const = 0;
      std::vector<std::pair<poly::SpectralPowerDistribution, float>> _output;
      const float x_width;
      const float y_width;
   };

}


#endif //POLY_ABSTRACTFILTER_H
