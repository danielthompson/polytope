//
// Created by Daniel on 19-Mar-18.
//

#include "TileRunner.h"

namespace Polytope {

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

   TileRunner::TileRunner(AbstractSampler *sampler, AbstractScene *scene, AbstractIntegrator *integrator,
                          AbstractFilm *film, const Polytope::Bounds bounds, unsigned int numSamples)
         : AbstractRunner(sampler, scene, integrator, film, numSamples, bounds) {

      _xLastTileWidth = getXLastTileWidth();
      _yLastTileWidth = getYLastTileWidth();

      _numXTiles = getXTiles();
      _numYTiles = getYTiles();
   }

   unsigned int TileRunner::getXLastTileWidth() const {

      unsigned int n = Bounds.x % _xTileWidth;
      if (n == 0)
         n = _xTileWidth;
      return n;
   }

   unsigned int TileRunner::getYLastTileWidth() const {
      unsigned int n = Bounds.x % _yTileWidth;
      if (n == 0)
         n = _yTileWidth;
      return n;
   }

   unsigned int TileRunner::getXTiles() const {
      if (Bounds.x % _xTileWidth > 0)
         return (Bounds.x / _xTileWidth) + 1;

      else
         return (Bounds.x / _xTileWidth);
   }

   unsigned int TileRunner::getYTiles() const {
      if (Bounds.y % _yTileWidth > 0)
         return (Bounds.y / _yTileWidth) + 1;

      else
         return (Bounds.y / _yTileWidth);
   }

   void TileRunner::getNextTile(Point2i &tile) {
      if (_xTilePointer < _numXTiles && _yTilePointer < _numYTiles) {
         tile.x = _xTilePointer;
         tile.y = _yTilePointer;

         _xTilePointer++;

         if (_xTilePointer >= _numXTiles) {
            _xTilePointer = 0;
            _yTilePointer++;
         }
      }
      else {
         int j = 0;
      }
   }
}