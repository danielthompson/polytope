//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_ABSTRACTRUNNER_H
#define POLY_ABSTRACTRUNNER_H

#include <memory>
#include <thread>
#include "../samplers/samplers.h"
#include "../scenes/scene.h"
#include "../integrators/abstract_integrator.h"
#include "../films/abstract_film.h"
#include "../structures/stats.h"

namespace poly {

   class runner {
   public:
      runner(
            std::unique_ptr<poly::abstract_sampler> sampler,
            std::shared_ptr<poly::scene> scene,
            std::shared_ptr<poly::abstract_integrator> integrator,
            std::unique_ptr<poly::abstract_film> film,
            const unsigned int sample_count,
            const poly::bounds bounds);

      void thread_entrypoint(int threadId) const;
      void trace(const int x, const int y) const;
      void output() const {
         Film->output();
      };
      std::thread spawn_thread(const int id) {
         return std::thread(&runner::thread_entrypoint, this, id);
      }

      ~runner();

      unsigned int sample_count;
      std::unique_ptr<poly::abstract_sampler> Sampler;
      std::unique_ptr<poly::abstract_film> Film;
      std::shared_ptr<poly::scene> Scene;
      std::shared_ptr<poly::abstract_integrator> integrator;

      const poly::bounds Bounds;
   };

}

#endif //POLY_ABSTRACTRUNNER_H
