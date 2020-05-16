//
// Created by daniel on 5/2/20.
//

#ifndef POLYTOPE_BVHNODE_H
#define POLYTOPE_BVHNODE_H

#include <vector>
#include "../structures/Vectors.h"
#include "../structures/BoundingBox.h"

namespace Polytope {
   class BVHNode {
   public:
      BVHNode *high = nullptr;
      BVHNode *low = nullptr;
      BVHNode *parent = nullptr;
      std::vector <Point3ui> faces;
      BoundingBox bbox;

      void ShrinkBoundingBox(const std::vector <Point> &vertices, const std::vector <Point3ui> &nodeFaces);
   };
}

#endif //POLYTOPE_BVHNODE_H
