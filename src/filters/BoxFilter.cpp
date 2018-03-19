//
// Created by Daniel on 16-Mar-18.
//

#include "BoxFilter.h"
#include "../structures/Point2.h"

namespace Polytope {

   void BoxFilter::AddSample(const Point2f &location, const Sample &sample) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      const unsigned int index = y * Bounds.x + x;

      int j = 0;
      if (x == 905 && y == 458) {
         j++;
         j--;
      }

      _data[index].push_back(sample);
   }

   void BoxFilter::AddSamples(const Point2f &location, const std::vector<Sample> &samples) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      const unsigned int index = y * Bounds.x + x;

      for (unsigned int i = 0; i < samples.size(); i++) {
         _data[index].push_back(samples[i]);
      }
   }

   SpectralPowerDistribution BoxFilter::Output(const Point2i &pixel) {

      int j = 0;
      if (pixel.x == 313 && pixel.y == 197) {
         j++;
         j--;
      }
      const unsigned int index = pixel.y * Bounds.x + pixel.x;

      const unsigned long numSamples = _data[index].size();

      SpectralPowerDistribution accum;

      for (unsigned long i = 0; i < numSamples; i++) {
         accum += _data[index][i].SpectralPowerDistribution;
      }

      const float divisor = 1.0f / numSamples;

      accum *= divisor;

      // fix to return SPD not Sample
      return accum;
   }


}