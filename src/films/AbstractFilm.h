//
// Created by Daniel Thompson on 3/6/18.
//

#ifndef POLYTOPE_ABSTRACTFILM_H
#define POLYTOPE_ABSTRACTFILM_H

#include "../structures/Sample.h"
#include "../structures/Point2.h"

namespace Polytope {

   class AbstractFilm {
   public:

      // constructors
      AbstractFilm(unsigned int width, unsigned int height) : Width(width), Height(height) { };

      // methods
      virtual void AddSample(const Point2f &location, const Sample &sample) = 0;
      virtual void Output() = 0;

      // destructors
      virtual ~AbstractFilm() = default;;

      // data
      unsigned int Width, Height;

   };

}


#endif //POLYTOPE_ABSTRACTFILM_H
