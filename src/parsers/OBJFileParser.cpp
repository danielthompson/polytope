//
// Created by Daniel on 15-Dec-19.
//

#include <sstream>
#include "OBJFileParser.h"
#include "../utilities/Common.h"
#include "../structures/BoundingBox.h"

namespace Polytope {
   void OBJFileParser::ParseFile(TriangleMesh* mesh, const std::string &filepath) const {

      std::unique_ptr<std::istream> stream = OpenStream(filepath);

      Polytope::Point min(FloatMax, FloatMax, FloatMax), max(-FloatMax, -FloatMax, -FloatMax);

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
                  // TODO error handling for non-existent face
                  const unsigned int v0 = stoui(word);
                  iss >> word;
                  const unsigned int v1 = stoui(word);
                  iss >> word;
                  const unsigned int v2 = stoui(word);
                  Point3ui face(v0, v1, v2);
                  mesh->Faces.push_back(face);

                  const Point p0 = mesh->Vertices[v0];
                  min.x = p0.x < min.x ? p0.x : min.x;
                  min.y = p0.y < min.y ? p0.y : min.y;
                  min.z = p0.z < min.z ? p0.z : min.z;

                  max.x = p0.x > max.x ? p0.x : max.x;
                  max.y = p0.y > max.y ? p0.y : max.y;
                  max.z = p0.z > max.z ? p0.z : max.z;

                  const Point p1 = mesh->Vertices[v1];

                  min.x = p1.x < min.x ? p1.x : min.x;
                  min.y = p1.y < min.y ? p1.y : min.y;
                  min.z = p1.z < min.z ? p1.z : min.z;

                  max.x = p1.x > max.x ? p1.x : max.x;
                  max.y = p1.y > max.y ? p1.y : max.y;
                  max.z = p1.z > max.z ? p1.z : max.z;

                  const Point p2 = mesh->Vertices[v2];
                  min.x = p2.x < min.x ? p2.x : min.x;
                  min.y = p2.y < min.y ? p2.y : min.y;
                  min.z = p2.z < min.z ? p2.z : min.z;

                  max.x = p2.x > max.x ? p2.x : max.x;
                  max.y = p2.y > max.y ? p2.y : max.y;
                  max.z = p2.z > max.z ? p2.z : max.z;
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

      float xlen = (max.x - min.x) * 0.5f;
      float ylen = (max.y - min.y) * 0.5f;
      float zlen = (max.z - min.z) * 0.5f;

      float xcentroid = min.x + xlen;
      float ycentroid = min.y + ylen;
      float zcentroid = min.z + zlen;

      float dx = -xcentroid;
      float dy = -ycentroid;
      float dz = -zcentroid;

      Transform t = Transform::Translate(dx, dy, dz);
      const auto newt = t * *(mesh->ObjectToWorld);

      mesh->ObjectToWorld = std::make_shared<Polytope::Transform>(t * *(mesh->ObjectToWorld));

      // object space bounding box with centroid at origin
      min.x += dx;
      min.y += dy;
      min.z += dz;

      max.x += dx;
      max.y += dy;
      max.z += dz;

      BoundingBox bb(min, max);

      mesh->ObjectToWorld->ApplyInPlace(bb);
      mesh->WorldToObject = std::make_shared<Polytope::Transform>(mesh->ObjectToWorld->Invert());
      mesh->BoundingBox->p0 = bb.p0;
      mesh->BoundingBox->p1 = bb.p1;
   }
}
