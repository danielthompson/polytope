//
// Created by daniel on 10/16/21.
//

#include "triangle_filter.h"

namespace poly {

   float calc_percent(const int x, const int y, const point2f& location) {
      const float x_min = std::min((float)x, (location.x - 0.5f));
      const float y_min = std::min((float)y, (location.y - 0.5f));
      const float x_max = std::max((float)(x + 1), (location.x + 0.5f));
      const float y_max = std::max((float)(y + 1), (location.y + 0.5f));
      
      const float x_overlap = 2.f - (x_max - x_min);
      const float y_overlap = 2.f - (y_max - y_min);
      if (x_overlap <= 0.f || y_overlap <= 0.f) {
         return 0.f;
      }
      const float area = x_overlap * y_overlap;
      
      return area;
   }
   
   void triangle_filter::add_sample(const point2f &location, const Sample &sample) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      for (int i = x - 1; i <= x + 1; i++) {
         if (i < 0 || i == Bounds.x)
            continue;
         for (int j = y - 1; j <= y + 1; j++) {
            if (j < 0 || j == Bounds.y)
               continue;
            const float percent = calc_percent(i, j, location);
            _output[j * Bounds.x + i] += (sample.SpectralPowerDistribution * percent);
         }
      }
   }

   void triangle_filter::add_samples(const point2f &location, const std::vector<Sample> &samples) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      const unsigned int index = y * Bounds.x + x;

      for (const auto &sample : samples) {
         _output[index] += sample.SpectralPowerDistribution;
      }
   }

   SpectralPowerDistribution triangle_filter::output(const point2i &pixel) const {
      const unsigned int index = pixel.y * Bounds.x + pixel.x;
      return _output[index];
   }

   void triangle_filter::pre_output() {
      const float samples_recip = 1.f / (float)num_samples;
      
      for (auto& spd : _output) {
         spd *= samples_recip;
      }
   }
}
