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
      PNGFilm(const Polytope::Bounds bounds, const std::string &filename, std::unique_ptr<AbstractFilter> filter) :
            AbstractFilm(bounds, std::move(filter)), Filename(std::move(filename))
             {
         //Data = std::vector<unsigned char>(size_t(bounds.x * bounds.y * 4));
      };

      // methods
      void AddSample(const Point2f &location, const Sample &sample) override;
      void Output() override;

      // data
      const std::string Filename;
   };

}


#endif //POLYTOPE_PNGFILM_H
