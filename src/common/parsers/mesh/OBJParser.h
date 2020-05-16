//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLYTOPE_OBJPARSER_H
#define POLYTOPE_OBJPARSER_H


#include "../AbstractFileParser.h"
#include "AbstractMeshParser.h"
#include "../../../cpu/shapes/abstract_mesh.h"

namespace Polytope {
   class OBJParser : public AbstractMeshParser {
   public:
      void ParseFile(AbstractMesh *mesh, const std::string &filepath) const override;
   };
}

#endif //POLYTOPE_OBJPARSER_H
