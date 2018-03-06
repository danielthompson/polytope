//
// Created by Daniel Thompson on 3/6/18.
//

#include <iostream>
#include "PNGFilm.h"
#include "../../lib/lodepng.h"

namespace Polytope {

   void PNGFilm::AddSample(Point2f location, Sample sample) {
      int index = int(location.x) * Width + int(location.y);

      int r = int(sample.SpectralPowerDistribution.r);
      int g = int(sample.SpectralPowerDistribution.g);
      int b = int(sample.SpectralPowerDistribution.b);
      int a = 0;

      Data[index + 0] = static_cast<unsigned char>(r);
      Data[index + 1] = static_cast<unsigned char>(g);
      Data[index + 2] = static_cast<unsigned char>(b);
      Data[index + 3] = static_cast<unsigned char>(a);
   }

   void PNGFilm::Output() {
      unsigned error = lodepng::encode(Filename.c_str(), Data, Width, Height);

      //if there's an error, display it
      if (error) {
            std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
      }
   }
}

