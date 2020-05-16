//
// Created by daniel on 5/2/20.
//

#include "BVHNode.h"

namespace Polytope {
   void BVHNode::ShrinkBoundingBox(const std::vector<Point> &vertices, const std::vector<Point3ui> &nodeFaces) {
      float minx = Polytope::FloatMax;
      float miny = Polytope::FloatMax;
      float minz = Polytope::FloatMax;

      float maxx = -Polytope::FloatMax;
      float maxy = -Polytope::FloatMax;
      float maxz = -Polytope::FloatMax;

      for (const Point3ui face : nodeFaces) {
         Point v0 = vertices[face.x];
         if (v0.x > maxx)
            maxx = v0.x;
         if (v0.x < minx)
            minx = v0.x;

         if (v0.y > maxy)
            maxy = v0.y;
         if (v0.y < miny)
            miny = v0.y;

         if (v0.z > maxz)
            maxz = v0.z;
         if (v0.z < minz)
            minz = v0.z;

         Point v1 = vertices[face.y];

         if (v1.x > maxx)
            maxx = v1.x;
         if (v1.x < minx)
            minx = v1.x;

         if (v1.y > maxy)
            maxy = v1.y;
         if (v1.y < miny)
            miny = v1.y;

         if (v1.z > maxz)
            maxz = v1.z;
         if (v1.z < minz)
            minz = v1.z;

         Point v2 = vertices[face.z];

         if (v2.x > maxx)
            maxx = v2.x;
         if (v2.x < minx)
            minx = v2.x;

         if (v2.y > maxy)
            maxy = v2.y;
         if (v2.y < miny)
            miny = v2.y;

         if (v2.z > maxz)
            maxz = v2.z;
         if (v2.z < minz)
            minz = v2.z;

      }

      bbox.p0.x = minx;
      bbox.p0.y = miny;
      bbox.p0.z = minz;

      bbox.p1.x = maxx;
      bbox.p1.y = maxy;
      bbox.p1.z = maxz;
   }
}
