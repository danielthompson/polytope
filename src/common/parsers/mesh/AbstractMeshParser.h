//
// Created by Daniel on 27-Dec-19.
//

#ifndef POLY_ABSTRACTMESHPARSER_H
#define POLY_ABSTRACTMESHPARSER_H

#include "../AbstractFileParser.h"
#include "../../../cpu/shapes/mesh.h"

namespace poly {
   class AbstractMeshParser : public AbstractFileParser {
   public:
      virtual void ParseFile(std::shared_ptr<poly::mesh_geometry> mesh, const std::string &filepath) const = 0;
      bool has_vertex_normals = false;
   };
}

#endif //POLY_ABSTRACTMESHPARSER_H
