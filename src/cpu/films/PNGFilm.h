//
// Created by Daniel Thompson on 3/6/18.
//

#ifndef POLY_PNGFILM_H
#define POLY_PNGFILM_H

#include <string>
#include <utility>
#include <vector>
#include "AbstractFilm.h"

namespace poly {

   class PNGFilm : public AbstractFilm {
   public:
      const std::string Filename;

      PNGFilm(
         const poly::Bounds bounds,
         std::string filename,
         std::unique_ptr<AbstractFilter> filter)
         : AbstractFilm(bounds, std::move(filter)),
            Filename(std::move(filename)) { };

      void AddSample(const Point2f &location, const Sample &sample) override;
      void Output() override;
   };
}

#endif //POLY_PNGFILM_H
