//
// Created by Daniel Thompson on 3/6/18.
//

#include <iostream>
#include <sstream>
#include "PNGFilm.h"
#include "../../../lib/lodepng.h"
#include "../../common/utilities/Common.h"
#include "../../common/utilities/GlobalDefines.h"

// TODO include linux / osx defines

namespace poly {

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

            const auto r = static_cast<unsigned char>(spd.r > 255 ? 255 : spd.r);
            const auto g = static_cast<unsigned char>(spd.g > 255 ? 255 : spd.g);
            const auto b = static_cast<unsigned char>(spd.b > 255 ? 255 : spd.b);
            const auto a = static_cast<unsigned char>(255);

            //std::cout << "writing index [" << index << "] for x [" << x << "], y [" << y << "]...";

            Data[index + 0] = r;
            Data[index + 1] = g;
            Data[index + 2] = b;
            Data[index + 3] = a;
         }
      }

      std::string cwd = GetCurrentWorkingDirectory();
      Log.WithTime("Outputting to file " + cwd + "/" + Filename + "...");

      unsigned error = lodepng::encode(Filename, Data, Bounds.x, Bounds.y);

      // if there's an error, display it
      if (error) {
         std::ostringstream oss;
         oss << "LodePNG encoding error (code " << error << "): " << lodepng_error_text(error);
         Log.WithTime(oss.str());
      }
   }
}
