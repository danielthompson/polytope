//
// Created by daniel on 5/21/20.
//

#ifndef POLY_CHECK_ERROR_H
#define POLY_CHECK_ERROR_H

#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"

#define cuda_check_error(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      Log.error("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
         exit(code);
   }
}

#define cuda_debug_printf(debug, s, ...) if (debug) printf("<<<%i, %i>>> " s, blockIdx.x, threadIdx.x, ##__VA_ARGS__)

#endif //POLY_CHECK_ERROR_H
