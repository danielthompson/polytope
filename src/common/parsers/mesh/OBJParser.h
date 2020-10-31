//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLY_OBJPARSER_H
#define POLY_OBJPARSER_H


#include "../AbstractFileParser.h"
#include "AbstractMeshParser.h"

namespace poly {
   class OBJParser : public AbstractMeshParser {
   public:
      void ParseFile(std::shared_ptr<poly::mesh_geometry> mesh, const std::string &filepath) const override;
   };
}

#endif //POLY_OBJPARSER_H
