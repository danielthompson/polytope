//
// Created by Daniel Thompson on 3/6/18.
//

#include <iostream>
#include "PNGFilm.h"
#include "../../lib/lodepng.h"

namespace Polytope {

   void PNGFilm::AddSample(const Point2f &location, const Sample &sample) {
//      int index = int(location.y) * Height + int(location.x);

      int x = int(location.x);
      int y = int(location.y);

      if (x == 0 && y == 200) {

      }

      int index = 4 * (y * Width + x);

      unsigned char r = static_cast<unsigned char>(sample.SpectralPowerDistribution.r);
      unsigned char g = static_cast<unsigned char>(sample.SpectralPowerDistribution.g);
      unsigned char b = static_cast<unsigned char>(sample.SpectralPowerDistribution.b);
      unsigned char a = static_cast<unsigned char>(255);

      //std::cout << "writing index [" << index << "] for x [" << x << "], y [" << y << "]...";

      Data[index + 0] = r;
      Data[index + 1] = g;
      Data[index + 2] = b;
      Data[index + 3] = a;

      //std::cout << " done" << std::endl;
   }

   void PNGFilm::Output() {
      unsigned error = lodepng::encode(Filename.c_str(), Data, Width, Height);

      //if there's an error, display it
      if (error) {
            std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
      }
   }
}

