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

            const int index = 4 * (y * width + x);
            const SpectralPowerDistribution spd = Filter->Output(Point2i(x, y));



            const unsigned char r = static_cast<unsigned char>(spd.r > 255 ? 255 : spd.r);
            const unsigned char g = static_cast<unsigned char>(spd.g > 255 ? 255 : spd.g);
            const unsigned char b = static_cast<unsigned char>(spd.b > 255 ? 255 : spd.b);
            const unsigned char a = static_cast<unsigned char>(255);

            //std::cout << "writing index [" << index << "] for x [" << x << "], y [" << y << "]...";

            Data[index + 0] = r;
            Data[index + 1] = g;
            Data[index + 2] = b;
            Data[index + 3] = a;
         }
      }

      unsigned error = lodepng::encode(Filename, Data, Bounds.x, Bounds.y);

      //if there's an error, display it
      if (error) {
            std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
      }
   }
}

