//
// Created by Daniel on 19-Mar-18.
//

#include <sstream>
#include <queue>
#include <mutex>
#include "runner.h"

extern struct poly::stats main_stats;
extern thread_local struct poly::stats thread_stats;
extern thread_local poly::random_number_generator rng;

namespace poly {

   namespace {
      constexpr unsigned int _tile_width_x = 2;
      constexpr unsigned int _tile_width_y = 2;
      std::vector<poly::point2i> _tiles;
      std::mutex _mutex;
      std::atomic<int> _tiles_current_index;
   }

   runner::runner(std::unique_ptr<poly::abstract_sampler> sampler,
                  std::shared_ptr<poly::scene> scene,
                  std::shared_ptr<poly::abstract_integrator> integrator,
                  std::unique_ptr<poly::abstract_film> film,
                  const unsigned int sample_count,
                  const poly::bounds bounds)
                                  : Sampler(std::move(sampler)),
                                    Scene(scene),
                                    integrator(integrator),
                                    Film(std::move(film)),
                                    sample_count(sample_count),
                                    Bounds(bounds) {

      for (int y = 0; y < bounds.y; y += _tile_width_y) {
         for (int x = 0; x < bounds.x; x += _tile_width_x) {
            _tiles.emplace_back(x, y);
         }
      }
      _tiles_current_index = 0;
   }
   
   void runner::thread_entrypoint(int threadId) const {

      const unsigned long stream_index = threadId;
      
      rng.increment_stream(stream_index);
      
      while (1) {
         
         int tile_index = _tiles_current_index++;
         if (tile_index >= _tiles.size())
            break;
         
         poly::point2i tile = _tiles[tile_index];
         
         for (unsigned int y = tile.y; y < tile.y + _tile_width_y && y < Bounds.y; y++) {
            for (unsigned int x = tile.x; x < tile.x + _tile_width_x && x < Bounds.x; x++) {
               trace(x, y);
            }
         }
      }
      
      // do stats
      std::lock_guard<std::mutex> lock(_mutex);
      LOG_DEBUG("Thread " << threadId << " set stream_index to " << stream_index << " and found stream_index at " << rng.stream_index);
      main_stats.add(thread_stats);
   }

   void runner::trace(const int x, const int y) const {

      poly::point2i pixel = {x, y};
      
      poly::point2f points[sample_count];

      Sampler->get_samples(points, sample_count, x, y);

      for (unsigned int i = 0; i < sample_count; i++) {

         poly::point2f sampleLocation = points[i];
         poly::ray ray = Scene->Camera->get_ray_for_pixel(sampleLocation);
         poly::Sample sample = integrator->get_sample(ray, 0, x, y);
         Film->add_sample(pixel, sampleLocation, sample);
      }
   }

   runner::~runner() {
      LOG_DEBUG("runner destructing");
   }
}
