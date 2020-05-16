//
// Created by Daniel Thompson on 3/6/18.
//

#ifndef POLYTOPE_PNGFILM_H
#define POLYTOPE_PNGFILM_H

#include <string>
#include <utility>
#include <vector>
#include "AbstractFilm.h"

namespace Polytope {

   class PNGFilm : public AbstractFilm {
   public:
      const std::string Filename;

      PNGFilm(
         const Polytope::Bounds bounds,
         std::string filename,
         std::unique_ptr<AbstractFilter> filter)
         : AbstractFilm(bounds, std::move(filter)),
            Filename(std::move(filename)) { };

      void AddSample(const Point2f &location, const Sample &sample) override;
      void Output() override;
   };
}

#endif //POLYTOPE_PNGFILM_H
