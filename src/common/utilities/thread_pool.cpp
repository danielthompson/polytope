//
// Created by daniel on 10/26/20.
//

#include "thread_pool.h"

std::mutex fprintf_mutex;

namespace poly {
   
//#define THREAD_POOL_DEBUG
#ifdef THREAD_POOL_DEBUG
#define thread_pool_printf(...) do { std::lock_guard<std::mutex> guard(fprintf_mutex); fprintf(stderr, __VA_ARGS__); } while (0)
#else
#define thread_pool_printf(...)
#endif

   thread_pool::thread_pool(const int num_threads) : num_threads(num_threads) {
      // create worker threads
      thread_pool_printf("ctor(): started\n");
      pool_state = starting;
      thread_states.reserve(num_threads);
      threads.reserve(num_threads);
      for (int i = 0; i < num_threads; i++) {
         thread_states.emplace_back(state::starting);
         threads.emplace_back(&thread_pool::run, this, i);
      }
      
      // block until all worker threads are ready for work
      while (true) {
         bool all_started = true;
         {
            std::lock_guard<std::mutex> guard(q_mutex);
            for (const thread_pool::state state : thread_states) {
               if (state == starting) {
                  all_started = false;
                  break;
               }
            }
         }
         
         if (all_started)
            break;
      }
      
      pool_state = ready;
      
      thread_pool_printf("ctor(): ended\n");
   }

   thread_pool::~thread_pool() {
      thread_pool_printf("~thread_pool(): started\n");
      if (pool_state != ending) {
         synchronize(true);
      }

      // join all threads
      for (int i = 0; i < num_threads; i++) {
         if (threads[i].joinable()) {
            thread_pool_printf("~thread_pool(): worker %i is joinable...\n", i);
            
            threads[i].join();
            thread_pool_printf("~thread_pool(): worker %i joined\n", i);
         }
         else
         {
            thread_pool_printf("~thread_pool(): worker %i is not joinable\n", i);
         }
      }
      thread_pool_printf("~thread_pool(): ended\n");
   }
   
   void thread_pool::synchronize(bool end) {
      thread_pool_printf("synchronize(): started, ending? %i\n", end);
      // set done flag
      if (end)
         pool_state = ending;
      else {
         pool_state = synchronizing;
      }

      // wake all threads
      ready_cvar.notify_all();

      std::unique_lock<std::mutex> q_lock(q_mutex, std::defer_lock);
      std::unique_lock<std::mutex> idle_lock(idle_mutex, std::defer_lock);
      while (true) {

         thread_pool_printf("synchronize(): locking q_lock...\n");
         q_lock.lock();
         thread_pool_printf("synchronize(): locked q_lock.\n");
         if (ready_q.empty()) {
            bool all_idle = true;
            thread_pool_printf("synchronize(): locking idle_lock...\n");
            idle_lock.lock();
            thread_pool_printf("synchronize(): locked idle_lock.\n");
            for (const auto &state : thread_states) {
               if (state != state::ready && state != state::ending) {
                  all_idle = false;
                  break;
               }
            }
            if (all_idle) {
               thread_pool_printf("synchronize(): all workers idle\n");
               idle_lock.unlock();
               q_lock.unlock();
               break;
            }
         }
         q_lock.unlock();
         thread_pool_printf("synchronize(): some workers running, waiting on idle cvar\n");
         if (idle_lock.owns_lock()) {
            thread_pool_printf("synchronize(): we already own idle_lock, so we don't re-lock it\n");
         }
         else {
            thread_pool_printf("synchronize(): we don't own idle_lock, so we lock it\n");
            idle_lock.lock();
         }
         idle_cvar.wait(idle_lock);
         thread_pool_printf("synchronize(): received idle cvar signal, checking again...\n");
         if (idle_lock.owns_lock()) {
            idle_lock.unlock();
            thread_pool_printf("synchronize(): we own idle_lock, so we unlock it\n");
         }
         else {
            thread_pool_printf("synchronize(): we don't own idle_lock, so we don't unlock it\n");   
         }
      }
      
      if (!end) {
         pool_state = ready;
         for (int i = 0; i < num_threads; i++) {
            thread_states[i] = ready;
         }
         thread_pool_printf("synchronize(): reset all workers\n");
      }
      
      thread_pool_printf("synchronize(): done\n");
   }
   
   void thread_pool::enqueue(const std::function<void()>& task) {
      thread_pool_printf("enqueue(): started\n");
      {
         std::lock_guard<std::mutex> lock(q_mutex);
         ready_q.push(task);
         thread_pool_printf("enqueue(): enqueued %i indices, signalling ready cvar...\n", indices);
         ready_cvar.notify_one();
      }
      
   }

   void thread_pool::run(const int thread_num) {

      std::unique_lock<std::mutex> q_lock(q_mutex, std::defer_lock);
      while (1) {
         // acquire ready Q mutex
         thread_pool_printf("thread_entrypoint(%i): Acquiring lock...\n", thread_num);
         q_lock.lock();
         thread_pool_printf("thread_entrypoint(%i): Lock acquired.\n", thread_num);

         // any tasks in Q?
         while (ready_q.empty()) {
            thread_pool_printf("thread_entrypoint(%i): Q empty, pool state %i\n", thread_num, pool_state);
            if (pool_state == ending) {
               q_lock.unlock();
               thread_pool_printf("thread_entrypoint(%i): Pool is ending, signalling idle cvar\n", thread_num);
               thread_states[thread_num] = ending;
               //std::lock_guard<std::mutex> guard(idle_mutex);
               idle_cvar.notify_one();
               thread_pool_printf("thread_entrypoint(%i): Returning\n", thread_num);
               return;
            }
            // wait on ready cvar
            thread_states[thread_num] = ready;
            {
               //std::lock_guard<std::mutex> guard(idle_mutex);
               thread_pool_printf("thread_entrypoint(%i): Signalling idle cvar...\n", thread_num);   
               idle_cvar.notify_one();
            }
            thread_pool_printf("thread_entrypoint(%i): Waiting on ready cvar...\n", thread_num);
            ready_cvar.wait(q_lock);
            thread_pool_printf("thread_entrypoint(%i): Woke on ready cvar\n", thread_num);
            if (!ready_q.empty()) {
               thread_pool_printf("thread_entrypoint(%i): Found work in Q\n", thread_num);
               break;
            }
         }

         std::function<void()> task = ready_q.front();
         
         ready_q.pop();
         thread_pool_printf("thread_entrypoint(%i): Popped task with %i indices, unlocking\n", thread_num, indices);
         thread_states[thread_num] = running;
         q_lock.unlock();

         thread_pool_printf("thread_entrypoint(%i): Running task...\n", thread_num);
         // process task
         task();

         // signal done
         thread_pool_printf("thread_entrypoint(%i): Task complete.\n", thread_num);
         // start over
      }
   }
}
