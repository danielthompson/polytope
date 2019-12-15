//
// Created by Daniel on 19-Mar-18.
//

#include "TileRunner.h"

namespace Polytope {

   namespace {
      constexpr unsigned int _xTileWidth = 32;
      constexpr unsigned int _yTileWidth = 32;

      unsigned int _xTilePointer = 0;
      unsigned int _yTilePointer = 0;

      unsigned int _xLastTileWidth;
      unsigned int _yLastTileWidth;

      unsigned int _numXTiles;
      unsigned int _numYTiles;

      std::mutex _mutex;

      unsigned int getXLastTileWidth(const Polytope::TileRunner &runner) {

         unsigned int n = runner.Bounds.x % _xTileWidth;
         if (n == 0)
            n = _xTileWidth;
         return n;
      }

      unsigned int getYLastTileWidth(const Polytope::TileRunner &runner) {
         unsigned int n = runner.Bounds.x % _yTileWidth;
         if (n == 0)
            n = _yTileWidth;
         return n;
      }

      unsigned int getXTiles(const Polytope::TileRunner &runner) {
         if (runner.Bounds.x % _xTileWidth > 0)
            return (runner.Bounds.x / _xTileWidth) + 1;

         else
            return (runner.Bounds.x / _xTileWidth);
      }

      unsigned int getYTiles(const Polytope::TileRunner &runner) {
         if (runner.Bounds.y % _yTileWidth > 0)
            return (runner.Bounds.y / _yTileWidth) + 1;

         else
            return (runner.Bounds.y / _yTileWidth);
      }

      void getNextTile(Point2i &tile) {

         //std::thread::id threadID = std::this_thread::get_id();
         //std::cout << "Thread " << threadID << " entering critical section..." << std::endl;
         //logger.logThread("Waiting for lock...");

         //auto outputStart = std::chrono::system_clock::now();

         std::lock_guard<std::mutex> lock(_mutex);

         //auto outputEnd = std::chrono::system_clock::now();
         //std::chrono::duration<double> outputtingElapsedSeconds = outputEnd - outputStart;
         //logger.logThread("Entered lock in " + std::to_string(outputtingElapsedSeconds.count()) + "s.");

         //std::cout << "Thread " << threadID << " entered critical section, " ;

         if (_xTilePointer < _numXTiles && _yTilePointer < _numYTiles) {
            //std::cout << " passed if check." << std::endl;
            tile.x = _xTilePointer;
            tile.y = _yTilePointer;

            _xTilePointer++;

            if (_xTilePointer >= _numXTiles) {
               _xTilePointer = 0;
               _yTilePointer++;
            }
         }
         //else {
            //std::cout << " failed if check." << std::endl;
         //};
         //logger.logThread("Left lock.");
      }
   }

   void TileRunner::Run() {
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

         for (int y = yMin; y < yMax; y++) {
            for (int x = xMin; x < xMax; x++) {
               Trace(x, y);
            }
         }

         tile.x = -1;
         tile.y = -1;
         getNextTile(tile);
      }
   }

   TileRunner::TileRunner(
         std::unique_ptr<AbstractSampler> sampler,
         AbstractScene *scene,
         std::unique_ptr<AbstractIntegrator> integrator,
         std::unique_ptr<AbstractFilm> film,
         const Polytope::Bounds bounds,
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
