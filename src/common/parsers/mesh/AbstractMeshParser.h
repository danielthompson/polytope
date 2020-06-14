//
// Created by Daniel on 27-Dec-19.
//

#ifndef POLY_ABSTRACTMESHPARSER_H
#define POLY_ABSTRACTMESHPARSER_H

#include "../AbstractFileParser.h"
#include "../../../cpu/shapes/mesh.h"

namespace poly {
   class AbstractMeshParser : public AbstractFileParser {
      virtual void ParseFile(Mesh *mesh, const std::string &filepath) const = 0;
   };
}

#endif //POLY_ABSTRACTMESHPARSER_H
