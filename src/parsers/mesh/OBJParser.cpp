//
// Created by Daniel on 15-Dec-19.
//

#include <sstream>
#include "OBJParser.h"
#include "../../utilities/Common.h"

namespace Polytope {
   void OBJParser::ParseFile(TriangleMesh* mesh, const std::string &filepath) const {

      std::unique_ptr<std::istream> stream = OpenStream(filepath);

      Polytope::Point min(FloatMax, FloatMax, FloatMax), max(-FloatMax, -FloatMax, -FloatMax);

      std::string line;
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
                  const float x = stof(word);
                  iss >> word;
                  const float y = stof(word);
                  iss >> word;
                  const float z = stof(word);
                  const Point p(x, y, z);
                  const Point worldPoint = mesh->ObjectToWorld->Apply(p);
                  mesh->Vertices.push_back(worldPoint);
                  continue;
               }
               case 'f': {
                  // parse vertex indices
                  iss >> word;
                  // TODO error handling for non-existent face
                  // obj faces are 1-indexed, but polytope is internally 0-indexed
                  const unsigned int v0 = stoui(word) - 1;
                  iss >> word;
                  const unsigned int v1 = stoui(word) - 1;
                  iss >> word;
                  const unsigned int v2 = stoui(word) - 1;
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

      mesh->BoundingBox->p0 = min;
      mesh->BoundingBox->p1 = max;
   }

   void OBJParser::ParseFile(TriangleMeshSOA *mesh, const std::string &filepath) const {

      std::unique_ptr<std::istream> stream = OpenStream(filepath);

      Polytope::Point min(FloatMax, FloatMax, FloatMax), max(-FloatMax, -FloatMax, -FloatMax);

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
                  const float x = stof(word);
                  iss >> word;
                  const float y = stof(word);
                  iss >> word;
                  const float z = stof(word);
                  p.x = x;
                  p.y = y;
                  p.z = z;
                  mesh->ObjectToWorld->ApplyInPlace(p);
                  mesh->x.push_back(p.x);
                  mesh->y.push_back(p.y);
                  mesh->z.push_back(p.z);
                  
                  mesh->num_vertices++;
                  
                  continue;
               }
               case 'f': {
                  // parse vertex indices
                  iss >> word;
                  // TODO error handling for non-existent face
                  // obj faces are 1-indexed, but polytope is internally 0-indexed
                  const unsigned int v0 = stoui(word) - 1;
                  iss >> word;
                  const unsigned int v1 = stoui(word) - 1;
                  iss >> word;
                  const unsigned int v2 = stoui(word) - 1;
                  mesh->fv0.push_back(v0);
                  mesh->fv1.push_back(v1);
                  mesh->fv2.push_back(v2);

                  mesh->num_faces++;
                  
                  {
                     const float p0x = mesh->x[v0];
                     min.x = p0x < min.x ? p0x : min.x;
                     max.x = p0x > max.x ? p0x : max.x;
                  }

                  {
                     const float p0y = mesh->y[v0];
                     min.y = p0y < min.y ? p0y : min.y;
                     max.y = p0y > max.y ? p0y : max.y;
                  }

                  {
                     const float p0z = mesh->z[v0];
                     min.z = p0z < min.z ? p0z : min.z;
                     max.z = p0z > max.z ? p0z : max.z;
                  }

                  {
                     const float p1x = mesh->x[v1];
                     min.x = p1x < min.x ? p1x : min.x;
                     max.x = p1x > max.x ? p1x : max.x;
                  }
                  
                  {
                     const float p1y = mesh->y[v1];
                     min.y = p1y < min.y ? p1y : min.y;
                     max.y = p1y > max.y ? p1y : max.y;
                  }
                  
                  {
                     const float p1z = mesh->z[v1];
                     min.z = p1z < min.z ? p1z : min.z;
                     max.z = p1z > max.z ? p1z : max.z;
                  }

                  {
                     const float p2x = mesh->x[v2];
                     min.x = p2x < min.x ? p2x : min.x;
                     max.x = p2x > max.x ? p2x : max.x;
                  }

                  {
                     const float p2y = mesh->y[v2];
                     min.y = p2y < min.y ? p2y : min.y;
                     max.y = p2y > max.y ? p2y : max.y;
                  }

                  {
                     const float p2z = mesh->z[v2];
                     min.z = p2z < min.z ? p2z : min.z;
                     max.z = p2z > max.z ? p2z : max.z;
                  }
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

      mesh->BoundingBox->p0 = min;
      mesh->BoundingBox->p1 = max;

      mesh->ExpandFaces();
   }
}
