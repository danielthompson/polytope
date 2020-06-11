//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLY_OBJPARSER_H
#define POLY_OBJPARSER_H


#include "../AbstractFileParser.h"
#include "AbstractMeshParser.h"
#include "../../../cpu/shapes/abstract_mesh.h"

namespace poly {
   template <class TMesh>
   class OBJParser : public AbstractMeshParser<TMesh> {
   public:
      void ParseFile(TMesh *mesh, const std::string &filepath) const override;
   };
}

#endif //POLY_OBJPARSER_H
