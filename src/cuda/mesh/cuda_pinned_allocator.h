//
// Created by daniel on 6/6/20.
//

#ifndef POLY_CUDA_PINNED_ALLOCATOR_H
#define POLY_CUDA_PINNED_ALLOCATOR_H
#include <cstdlib>
#include <new>
#include <limits>
#include <cuda_runtime.h>
#include <cstdio>

namespace poly {

   template <class T>
   struct cuda_pinned_allocator
   {
      typedef T value_type;

      cuda_pinned_allocator () = default;
      template <class U> constexpr cuda_pinned_allocator (const cuda_pinned_allocator <U>&) noexcept {}

      T* allocate(std::size_t n) {
         if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

         T* pointer;
         
         // TODO investigate performance of other flags
         cudaError_t error = cudaHostAlloc(&pointer, n * sizeof(T), cudaHostAllocDefault);
            
         if (error == cudaSuccess) {
            return pointer;
         }

         throw std::bad_alloc();
      }
      void deallocate(T* p, std::size_t) noexcept { 
         cudaError_t error = cudaFreeHost(p);
         if (error != cudaSuccess) {
            fprintf(stderr, "couldn't free pinned memory. shit :/");
         }
      }
   };

   template <class T, class U>
   bool operator==(const cuda_pinned_allocator <T>&, const cuda_pinned_allocator <U>&) { return true; }
   template <class T, class U>
   bool operator!=(const cuda_pinned_allocator <T>&, const cuda_pinned_allocator <U>&) { return false; }
}

#endif //POLY_CUDA_PINNED_ALLOCATOR_H
