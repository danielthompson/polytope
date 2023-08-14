//
// Created by daniel on 10/26/20.
//

#ifndef POLYTOPE_THREAD_POOL_H
#define POLYTOPE_THREAD_POOL_H

#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <thread>

namespace poly {
   /**
    * Simple thread pool.
    */
   class thread_pool {
   public:
      /**
       * Create a new thread pool with an empty task queue.
       * @param num_threads The number of worker threads to start in the pool.
       */
      explicit thread_pool(int num_threads);
      
      /**
       * Destroy the thread pool. Blocks until the task queue is empty and all worker threads are idle.
       */
      ~thread_pool();
      
      /**
       * Enqueue the given task to the task pool.
       * @param task The task to enqueue to the pool.
       */
      void enqueue(const std::function<void()>& task);
      
      /**
       * Blocks the _calling_ thread until the task queue is empty and all worker threads are idle. Note that this 
       * doesn't prevent threads other than the caller (e.g. worker threads) from continuing to enqueue tasks to the
       * pool. If that happens, this method will continue to block until all child tasks are complete. 
       */
      void synchronize(bool end = false);

   private:
      void run(int thread_num);
      bool done = false;
      int num_threads;
      std::queue<std::function<void()>> ready_q;
      std::vector<std::thread> threads;
      
      enum state {
         starting = 0,
         ready = 1,
         running = 2,
         synchronizing = 3,
         ending = 4
      };
      
      std::vector<state> thread_states;
      state pool_state;
      std::mutex q_mutex;
      std::condition_variable ready_cvar;
      std::mutex idle_mutex;
      std::condition_variable idle_cvar;
   };
}


#endif //POLYTOPE_THREAD_POOL_H
