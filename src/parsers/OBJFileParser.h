//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLYTOPE_OBJFILEPARSER_H
#define POLYTOPE_OBJFILEPARSER_H

#include "../shapes/TriangleMesh.h"
#include "AbstractFileParser.h"

namespace Polytope {

   class OBJFileParser : public AbstractFileParser {
   public:
      std::unique_ptr<TriangleMesh> ParseFile(const std::string &filepath);
   };

}

#endif //POLYTOPE_OBJFILEPARSER_H
