//
// Created by Daniel on 16-Mar-18.
//

#include <sstream>
#include "box_filter.h"
#include "../../common/structures/point2.h"
#include "../../common/utilities/Common.h"

namespace poly {

   void box_filter::add_sample(const poly::point2f &location, const poly::Sample &sample) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      const int index = y * Bounds.x + x;

      const auto vectorSize = _data.size();

      if (vectorSize <= index) {
         LOG_DEBUG("Attempting to write a sample for (" << x << ", " << y << ") to index " <<  index << " but size is " << vectorSize << "... :/");
      }

      _data[index].push_back(sample);
   }

   void box_filter::add_samples(const poly::point2f &location, const std::vector<poly::Sample> &samples) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      const unsigned int index = y * Bounds.x + x;

      // std::lock_guard<std::mutex> lock(_mutex);

      for (const auto &sample : samples) {
         Sample clamped = sample;
         if (clamped.SpectralPowerDistribution.r > 255)  clamped.SpectralPowerDistribution.r = 255;
         if (clamped.SpectralPowerDistribution.g > 255)  clamped.SpectralPowerDistribution.g = 255;
         if (clamped.SpectralPowerDistribution.b > 255)  clamped.SpectralPowerDistribution.b = 255;
         _data[index].push_back(clamped);
      }
   }

   poly::SpectralPowerDistribution box_filter::output(const poly::point2i &pixel) const {
      const unsigned int index = pixel.y * Bounds.x + pixel.x;

      const unsigned long numSamples = _data[index].size();

      poly::SpectralPowerDistribution accum;

      for (unsigned long i = 0; i < numSamples; i++) {
         accum += _data[index][i].SpectralPowerDistribution;
      }

      const float divisor = 1.0f / (float)numSamples;

      accum *= divisor;

      // fix to return SPD not Sample
      return accum;
   }
}
