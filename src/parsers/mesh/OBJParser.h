//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLYTOPE_OBJPARSER_H
#define POLYTOPE_OBJPARSER_H

#include "../../shapes/triangle.h"
#include "../AbstractFileParser.h"
#include "AbstractMeshParser.h"

namespace Polytope {
   class OBJParser : public AbstractMeshParser {
   public:
      void ParseFile(TriangleMesh* mesh, const std::string &filepath) const override;
      void ParseFile(TriangleMeshSOA* mesh, const std::string &filepath) const override;
   };
}

#endif //POLYTOPE_OBJPARSER_H
