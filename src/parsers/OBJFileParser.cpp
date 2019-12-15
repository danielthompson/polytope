//
// Created by Daniel on 15-Dec-19.
//

#include <sstream>
#include "OBJFileParser.h"
#include "../utilities/Common.h"

namespace Polytope {
   std::unique_ptr<TriangleMesh> OBJFileParser::ParseFile(const std::string &filepath) {

      std::unique_ptr<std::istream> stream = OpenStream(filepath);

      Transform identity = Transform();
      std::unique_ptr<TriangleMesh> mesh = std::make_unique<TriangleMesh>(identity, nullptr);

      std::string line;
      while (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (iss >> word) {
            if (word.find('#') == 0)
               break;
            else if (word == "v") {
               // parse vertex coordinates
               iss >> word;
               float x = stof(word);
               iss >> word;
               float y = stof(word);
               iss >> word;
               float z = stof(word);
               Point p(x, y, z);
               mesh->Vertices.push_back(p);
               continue;
            }
            else if (word == "f") {
               // parse vertex indices
               iss >> word;
               const unsigned int v0 = stoui(word);
               iss >> word;
               const unsigned int v1 = stoui(word);
               iss >> word;
               const unsigned int v2 = stoui(word);
               Point3ui face(v0, v1, v2);
               mesh->Faces.push_back(face);
               continue;
            }
         }
      }

      return mesh;
   }
}
