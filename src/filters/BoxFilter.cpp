//
// Created by Daniel on 16-Mar-18.
//

#include <sstream>
#include "BoxFilter.h"
#include "../structures/Point2.h"
#include "../utilities/Common.h"

namespace Polytope {

   //std::mutex BoxFilter::_mutex;

   void BoxFilter::AddSample(const Point2f &location, const Sample &sample) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      const int index = y * Bounds.x + x;

      // std::lock_guard<std::mutex> lock(_mutex);

      const auto vectorSize = _data.size();

      if (vectorSize <= index) {
         std::ostringstream oss;
         oss << "Attempting to write a sample for (" << x << ", " << y << ") to index " <<  index << " but size is " << vectorSize << "... :/";
         Log.WithTime(oss.str());
      }

      _data[index].push_back(sample);
   }

   void BoxFilter::AddSamples(const Point2f &location, const std::vector<Sample> &samples) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      const unsigned int index = y * Bounds.x + x;

      // std::lock_guard<std::mutex> lock(_mutex);

      for (const auto &sample : samples) {
         _data[index].push_back(sample);
      }
   }

   SpectralPowerDistribution BoxFilter::Output(const Point2i &pixel) {

      if (pixel.x == 487 && pixel.y == 299) {
         int j = 0;
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

   void BoxFilter::SetSamples(const unsigned int samples) {
      for (int x = 0; x < Bounds.x; x++) {
         for (int y = 0; y < Bounds.y; y++) {
            const unsigned int index = y * Bounds.x + x;
            _data[index].reserve(samples);
         }
      }
   }
}
