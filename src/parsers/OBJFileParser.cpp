//
// Created by Daniel on 15-Dec-19.
//

#include <sstream>
#include "OBJFileParser.h"
#include "../utilities/Common.h"

namespace Polytope {
   void OBJFileParser::ParseFile(const std::shared_ptr<TriangleMesh>& mesh, const std::string &filepath) const {

      std::unique_ptr<std::istream> stream = OpenStream(filepath);

      std::string line;
      while (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (iss >> word) {
            char firstChar = word[0];
            switch (firstChar) {
               case '#': {
                  continue;
               }
               case 'v': {
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
               case 'f': {
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
               default: {
                  std::string error = "OBJ Parser: Ignoring line with unimplemented first char [";
                  error += firstChar;
                  error += "].";
                  Log.WithTime(error);
               }
            }
         }
      }
   }
}
