//
// Created by Daniel on 19-Mar-18.
//

#include <sstream>
#include "TileRunner.h"
#include "../structures/stats.h"
#include "../../common/utilities/Common.h"

extern struct poly::stats main_stats;
extern thread_local struct poly::stats thread_stats;

namespace poly {

   namespace {
      constexpr unsigned int _xTileWidth = 2;
      constexpr unsigned int _yTileWidth = 2;

      unsigned int _xTilePointer = 0;
      unsigned int _yTilePointer = 0;

      unsigned int _xLastTileWidth;
      unsigned int _yLastTileWidth;

      unsigned int _numXTiles;
      unsigned int _numYTiles;

      std::mutex _mutex;

      unsigned int getXLastTileWidth(const poly::TileRunner &runner) {

         unsigned int n = runner.Bounds.x % _xTileWidth;
         if (n == 0)
            n = _xTileWidth;
         return n;
      }

      unsigned int getYLastTileWidth(const poly::TileRunner &runner) {
         unsigned int n = runner.Bounds.x % _yTileWidth;
         if (n == 0)
            n = _yTileWidth;
         return n;
      }

      unsigned int getXTiles(const poly::TileRunner &runner) {
         if (runner.Bounds.x % _xTileWidth > 0)
            return (runner.Bounds.x / _xTileWidth) + 1;

         else
            return (runner.Bounds.x / _xTileWidth);
      }

      unsigned int getYTiles(const poly::TileRunner &runner) {
         if (runner.Bounds.y % _yTileWidth > 0)
            return (runner.Bounds.y / _yTileWidth) + 1;

         else
            return (runner.Bounds.y / _yTileWidth);
      }

      void getNextTile(Point2i &tile) {
         std::lock_guard<std::mutex> lock(_mutex);
         if (_xTilePointer < _numXTiles && _yTilePointer < _numYTiles) {
            tile.x = _xTilePointer;
            tile.y = _yTilePointer;

            _xTilePointer++;

            if (_xTilePointer >= _numXTiles) {
               _xTilePointer = 0;
               _yTilePointer++;
            }
         }
      }
   }

   void TileRunner::Run(int threadId) {
      Point2i tile(-1, -1);

      getNextTile(tile);

      while (tile.x != -1) {

         const unsigned int xMin = tile.x * _xTileWidth;
         const unsigned int yMin = tile.y * _yTileWidth;

         unsigned int xMax = (tile.x + 1) * _xTileWidth;
         unsigned int yMax = (tile.y + 1) * _yTileWidth;

         if (tile.x == _numXTiles - 1)
            xMax = tile.x * _xTileWidth + _xLastTileWidth;

         if (tile.y == _numYTiles - 1)
            yMax = tile.y * _yTileWidth + _yLastTileWidth;

         xMax = std::min(xMax, Bounds.x);
         yMax = std::min(yMax, Bounds.y);
         
         for (unsigned int y = yMin; y < yMax; y++) {
            for (unsigned int x = xMin; x < xMax; x++) {
               Trace(x, y);
            }
         }

//         std::ostringstream oss;
//         oss << "Thread " << threadId << ": Finished tile [" << xMin << ", " << yMin << "] x [" << xMax << ", " << yMax << "]";
//         Log.WithTime(oss.str());

         tile.x = -1;
         tile.y = -1;
         getNextTile(tile);
      }
      
      // do stats
      std::lock_guard<std::mutex> lock(_mutex);
      main_stats.add(thread_stats);
   }

   TileRunner::TileRunner(
         std::unique_ptr<AbstractSampler> sampler,
         poly::Scene *scene,
         std::unique_ptr<AbstractIntegrator> integrator,
         std::unique_ptr<AbstractFilm> film,
         const poly::Bounds bounds,
         unsigned int numSamples)
         : AbstractRunner(
            std::move(sampler),
            scene,
            std::move(integrator),
            std::move(film),
            numSamples,
            bounds) {

      _xLastTileWidth = getXLastTileWidth(*this);
      _yLastTileWidth = getYLastTileWidth(*this);

      _numXTiles = getXTiles(*this);
      _numYTiles = getYTiles(*this);
   }
}
