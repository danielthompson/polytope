//
// Created by Daniel on 19-Mar-18.
//

#include <sstream>
#include <queue>
#include <mutex>
#include "AbstractRunner.h"

extern struct poly::stats main_stats;
extern thread_local struct poly::stats thread_stats;

namespace poly {

   namespace {
      constexpr unsigned int _tile_width_x = 2;
      constexpr unsigned int _tile_width_y = 2;
      std::vector<poly::Point2i> _tiles;
      std::mutex _mutex;
      std::atomic<int> _tiles_current_index;
   }

   AbstractRunner::AbstractRunner(std::unique_ptr<AbstractSampler> sampler, 
                                  std::shared_ptr<poly::scene> scene,
                                  std::shared_ptr<poly::AbstractIntegrator> integrator,
                                  std::unique_ptr<AbstractFilm> film, 
                                  const unsigned int numSamples,
                                  const poly::Bounds bounds)
                                  : Sampler(std::move(sampler)),
                                    Scene(scene),
                                    Integrator(integrator),
                                    Film(std::move(film)),
                                    NumSamples(numSamples),
                                    Bounds(bounds) {

      for (int y = 0; y < bounds.y; y += _tile_width_y) {
         for (int x = 0; x < bounds.x; x += _tile_width_x) {
            _tiles.emplace_back(x, y);
         }
      }
      _tiles_current_index = 0;
   }
   
   void AbstractRunner::Run(int threadId) const {

      while (1) {
         
         int tile_index = _tiles_current_index++;
         if (tile_index >= _tiles.size())
            break;
         
         poly::Point2i tile = _tiles[tile_index];
         
         for (unsigned int y = tile.y; y < tile.y + _tile_width_y && y < Bounds.y; y++) {
            for (unsigned int x = tile.x; x < tile.x + _tile_width_x && x < Bounds.x; x++) {
               Trace(x, y);
            }
         }
      }
      
      // do stats
      std::lock_guard<std::mutex> lock(_mutex);
      main_stats.add(thread_stats);
   }

   void AbstractRunner::Trace(const int x, const int y) const {

      Point2f points[NumSamples];

      Sampler->GetSamples(points, NumSamples, x, y);

      for (unsigned int i = 0; i < NumSamples; i++) {

         Point2f sampleLocation = points[i];
         Ray ray = Scene->Camera->get_ray_for_pixel(sampleLocation);
         Sample sample = Integrator->GetSample(ray, 0, x, y);
         Film->AddSample(sampleLocation, sample);
      }
   }
}
