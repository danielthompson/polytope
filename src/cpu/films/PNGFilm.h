//
// Created by Daniel Thompson on 3/6/18.
//

#ifndef POLY_PNGFILM_H
#define POLY_PNGFILM_H

#include <string>
#include <utility>
#include <vector>
#include "abstract_film.h"

namespace poly {

   class PNGFilm : public abstract_film {
   public:
      const std::string Filename;

      PNGFilm(
         const poly::bounds bounds,
         std::string filename,
         std::unique_ptr<AbstractFilter> filter)
         : abstract_film(bounds, std::move(filter)),
           Filename(std::move(filename)) { };

      void AddSample(const point2f &location, const Sample &sample) override;
      void Output() override;
   };
}

#endif //POLY_PNGFILM_H
