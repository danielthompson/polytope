//
// Created by Daniel on 19-Mar-18.
//

#ifndef POLYTOPE_TILERUNNER_H
#define POLYTOPE_TILERUNNER_H

#include <mutex>
#include "AbstractRunner.h"
#include "../utilities/Logger.h"

namespace Polytope {

   class TileRunner : public AbstractRunner {
   public:
      explicit TileRunner(
            std::unique_ptr<AbstractSampler> sampler,
            AbstractScene *scene,
            std::unique_ptr<AbstractIntegrator> integrator,
            std::unique_ptr<AbstractFilm> film,
            Polytope::Bounds bounds,
            unsigned int numSamples);

      void Run() override;

   private:

      unsigned int getXLastTileWidth() const;
      unsigned int getYLastTileWidth() const;

      unsigned int getXTiles() const;
      unsigned int getYTiles() const;

      void getNextTile(Point2i &tile);

      const unsigned int _xTileWidth = 32;
      const unsigned int _yTileWidth = 32;

      unsigned int _xTilePointer = 0;
      unsigned int _yTilePointer = 0;

      unsigned int _xLastTileWidth;
      unsigned int _yLastTileWidth;

      unsigned int _numXTiles;
      unsigned int _numYTiles;

      static std::mutex _mutex;
      Logger logger = Logger();

   };

}


#endif //POLYTOPE_TILERUNNER_H
