//
// Created by Daniel on 27-Dec-19.
//

#ifndef POLYTOPE_ABSTRACTMESHPARSER_H
#define POLYTOPE_ABSTRACTMESHPARSER_H

#include "../AbstractFileParser.h"
#include "../../../cpu/shapes/abstract_mesh.h"

namespace Polytope {
   class AbstractMeshParser : public AbstractFileParser {
      virtual void ParseFile(AbstractMesh *mesh, const std::string &filepath) const = 0;
   };
}

#endif //POLYTOPE_ABSTRACTMESHPARSER_H
