//
// Created by daniel on 10/16/21.
//

#ifndef POLYTOPE_TRIANGLE_FILTER_H
#define POLYTOPE_TRIANGLE_FILTER_H


#include "abstract_filter.h"

namespace poly {

   class triangle_filter : public abstract_filter {
   public:

      explicit triangle_filter(const poly::bounds &bounds, const int num_samples)
            : abstract_filter(bounds),
              _output(std::vector<poly::SpectralPowerDistribution>(bounds.x * bounds.y)),
              num_samples(num_samples) { }

      SpectralPowerDistribution output(const point2i &pixel) const override;
      void add_sample(const point2f &location, const Sample &sample) override;
      void add_samples(const point2f &location, const std::vector<Sample> &samples) override;

      void pre_output() override;
      
   private:
      std::vector<poly::SpectralPowerDistribution> _output; 
      int num_samples;
   };
}


#endif //POLYTOPE_TRIANGLE_FILTER_H
