//
// Created by Daniel on 27-Dec-19.
//

#ifndef POLYTOPE_PLYPARSER_H
#define POLYTOPE_PLYPARSER_H

#include "../AbstractFileParser.h"
#include "AbstractMeshParser.h"
#include "../../shapes/triangle.h"

namespace Polytope {
   class PLYParser : public AbstractMeshParser {
   public:
      void ParseFile(TriangleMesh *mesh, const std::string &filepath) const override;
      void ParseFile(TriangleMeshSOA* mesh, const std::string &filepath) const override;
   };
}

#endif //POLYTOPE_PLYPARSER_H
