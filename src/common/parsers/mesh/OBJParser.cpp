//
// Created by Daniel on 15-Dec-19.
//

#include <sstream>
#include "OBJParser.h"
#include "../../utilities/Common.h"
#include "../../../cpu/shapes/mesh.h"

namespace poly {
   void OBJParser::ParseFile(Mesh *mesh, const std::string &filepath) const {

      std::unique_ptr<std::istream> stream = AbstractFileParser::open_ascii_stream(filepath);
      std::string line;
      Point p;
      while (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (iss >> word) {
            const char firstChar = word[0];
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
                  p.x = x;
                  p.y = y;
                  p.z = z;
                  mesh->add_vertex(x, y, z);
                  
                  continue;
               }
               case 'f': {
                  // parse vertex indices
                  iss >> word;
                  // TODO error handling for non-existent face
                  // obj faces are 1-indexed, but polytope is internally 0-indexed
                  const unsigned int v0 = AbstractFileParser::stoui(word) - 1;
                  iss >> word;
                  const unsigned int v1 = AbstractFileParser::stoui(word) - 1;
                  iss >> word;
                  const unsigned int v2 = AbstractFileParser::stoui(word) - 1;
                  mesh->add_packed_face(v0, v1, v2);
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

      mesh->unpack_faces();
   }
}
