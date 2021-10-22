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

   class png_film : public abstract_film {
   public:
      const std::string Filename;

      png_film(
         const poly::bounds bounds,
         std::string filename,
         std::unique_ptr<poly::abstract_filter> filter)
         : abstract_film(bounds, std::move(filter)),
           Filename(std::move(filename)) { };

      void add_sample(const poly::point2i& pixel, const poly::point2f& location, const poly::Sample& sample) override;
      void output() override;
   };
}

#endif //POLY_PNGFILM_H
