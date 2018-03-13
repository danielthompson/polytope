//
// Created by Daniel Thompson on 3/6/18.
//

#ifndef POLYTOPE_PNGFILM_H
#define POLYTOPE_PNGFILM_H


#include <string>
#include <vector>
#include "AbstractFilm.h"

namespace Polytope {

   class PNGFilm : public AbstractFilm {
   public:

      // constructors
      PNGFilm(unsigned int width, unsigned int height, const std::string &filename) :
            AbstractFilm(width, height), Filename(std::move(filename)) {
         Data = std::vector<unsigned char>(size_t(width * height * 4));
      };

      // methods
      void AddSample(const Point2f &location, const Sample &sample) override;
      void Output() override;

      // data
      std::string Filename;
      std::vector<unsigned char> Data;
   };

}


#endif //POLYTOPE_PNGFILM_H
