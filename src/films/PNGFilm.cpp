//
// Created by Daniel Thompson on 3/6/18.
//

#include <iostream>
#include "PNGFilm.h"
#include "../../lib/lodepng.h"

namespace Polytope {

   void PNGFilm::AddSample(const Point2f &location, const Sample &sample) {
      Filter->AddSample(location, sample);
   }

   void PNGFilm::Output() {

      std::vector<unsigned char> Data(size_t(Bounds.x * Bounds.y * 4));

      const unsigned int width = Bounds.x;
      const unsigned int height = Bounds.y;

      for (int x = 0; x < width; x++) {
         for (int y = 0; y < height; y++) {
            int index = 4 * (y * width + x);

            Sample sample = Filter->Output(Point2i(x, y));

            unsigned char r = static_cast<unsigned char>(sample.SpectralPowerDistribution.r);
            unsigned char g = static_cast<unsigned char>(sample.SpectralPowerDistribution.g);
            unsigned char b = static_cast<unsigned char>(sample.SpectralPowerDistribution.b);
            unsigned char a = static_cast<unsigned char>(255);

            //std::cout << "writing index [" << index << "] for x [" << x << "], y [" << y << "]...";

            Data[index + 0] = r;
            Data[index + 1] = g;
            Data[index + 2] = b;
            Data[index + 3] = a;
         }
      }

      unsigned error = lodepng::encode(Filename.c_str(), Data, Bounds.x, Bounds.y);

      //if there's an error, display it
      if (error) {
            std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
      }
   }
}

