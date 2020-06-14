//
// Created by daniel on 6/7/20.
//

#ifndef POLY_CUDA_MESH_SOA_H
#define POLY_CUDA_MESH_SOA_H

#include <vector>
#include "cuda_pinned_allocator.h"

namespace poly {

   class cuda_mesh_soa {
   public:
      std::vector<float, cuda_pinned_allocator<float>> x, y, z;
      
   };
}


#endif //POLY_CUDA_MESH_SOA_H
