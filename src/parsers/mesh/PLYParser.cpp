//
// Created by Daniel on 27-Dec-19.
//

#include <sstream>
#include "PLYParser.h"
#include "../../utilities/Common.h"

namespace Polytope {
   void PLYParser::ParseFile(TriangleMesh *mesh, const std::string &filepath) const {
      std::unique_ptr<std::istream> stream = OpenStream(filepath);

      Polytope::Point min(FloatMax, FloatMax, FloatMax), max(-FloatMax, -FloatMax, -FloatMax);

      std::string line;

      // ply header
      if (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (!(iss >> word) || word != "ply") {
            Log.WithTime("File missing PLY header :/");
            return;
         }
      }
      else {
         Log.WithTime("Read error when trying to read PLY header :/");
         return;
      }

      // format
      if (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (!(iss >> word) || word != "format") {
            Log.WithTime("File missing format header");
            return;
         }
         if (!(iss >> word) || word != "ascii") {
            Log.WithTime("Unsupported format");
            return;
         }
         if (!(iss >> word) || word != "1.0") {
            Log.WithTime("Unsupported format version");
            return;
         }
      }
      else {
         Log.WithTime("Read error when trying to read PLY format :/");
         return;
      }

      // rest of header

      bool inVertex = false;
      bool inFace = false;
      int numVertices = -1;
      int numFaces = -1;
      while (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);
         iss >> word;
         if (word == "end_header") {
            break;
         }
         else if (word == "element") {
            inVertex = false;
            inFace = false;
            iss >> word;
            if (word == "vertex") {
               inVertex = true;
               iss >> word;
               numVertices = stoui(word);
            }
            else if (word == "face") {
               inFace = true;
               iss >> word;
               numFaces = stoui(word);
            }
         }
         else if (word == "property") {
            if (inVertex) {
               iss >> word;
               if (word != "float32" && word != "float") {
                  Log.WithTime("unknown property datatype :/");
                  return;
               }
               iss >> word;
               if (!(word == "x" || word == "y" || word == "z")) {
                  Log.WithTime("unknown property name :/");
                  //return;
               }
            }
            else if (inFace) {

            }
            else {
               Log.WithTime("property outside of element :/");
               return;
            }
         }
      }

      if (numVertices <= 0) {
         Log.WithTime("header contains no vertex info :/");
         return;
      }

      if (numFaces <= 0) {
         Log.WithTime("header contains no face info :/");
         return;
      }

      // data - vertices

      for (int i = 0; i < numVertices; i++) {
         if (!getline(*stream, line)) {
            Log.WithTime("Failed to read line in vertices :/");
            return;
         }

         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         iss >> word;
         const float x = stof(word);
         iss >> word;
         const float y = stof(word);
         iss >> word;
         const float z = stof(word);
         const Point p(x, y, z);
         const Point worldPoint = mesh->ObjectToWorld->Apply(p);
         mesh->Vertices.push_back(worldPoint);
      }

      std::ostringstream str;
      str << "Parsed " << mesh->Vertices.size() << " vertices.";
      Log.WithTime(str.str());
      str.str("");

      // data - faces

      for (int i = 0; i < numFaces; i++) {
         if (!getline(*stream, line)) {
            Log.WithTime("Failed to read line in faces :/");
            return;
         }

         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         // parse vertex indices

         iss >> word;
         int numVertexIndices = stoui(word);
         if (numVertexIndices != 3) {
            Log.WithTime("Face has too many vertex indices :/");
            return;
         }

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

         bool debug = false;
         if (min.y == 0)
            debug = true;
      }

      str << "Parsed " << mesh->Faces.size() << " faces.";
      Log.WithTime(str.str());

      mesh->BoundingBox->p0 = min;
      mesh->BoundingBox->p1 = max;
   }

   void PLYParser::ParseFile(TriangleMeshSOA *mesh, const std::string &filepath) const {
      std::unique_ptr<std::istream> stream = OpenStream(filepath);

      Polytope::Point min(FloatMax, FloatMax, FloatMax), max(-FloatMax, -FloatMax, -FloatMax);

      std::string line;

      // ply header
      if (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (!(iss >> word) || word != "ply") {
            Log.WithTime("File missing PLY header :/");
            return;
         }
      }
      else {
         Log.WithTime("Read error when trying to read PLY header :/");
         return;
      }

      // format
      if (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (!(iss >> word) || word != "format") {
            Log.WithTime("File missing format header");
            return;
         }
         if (!(iss >> word) || word != "ascii") {
            Log.WithTime("Unsupported format");
            return;
         }
         if (!(iss >> word) || word != "1.0") {
            Log.WithTime("Unsupported format version");
            return;
         }
      }
      else {
         Log.WithTime("Read error when trying to read PLY format :/");
         return;
      }

      // rest of header

      bool inVertex = false;
      bool inFace = false;
      int numVertices = -1;
      int numFaces = -1;
      while (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);
         iss >> word;
         if (word == "end_header") {
            break;
         }
         else if (word == "element") {
            inVertex = false;
            inFace = false;
            iss >> word;
            if (word == "vertex") {
               inVertex = true;
               iss >> word;
               numVertices = stoui(word);
            }
            else if (word == "face") {
               inFace = true;
               iss >> word;
               numFaces = stoui(word);
            }
         }
         else if (word == "property") {
            if (inVertex) {
               iss >> word;
               if (word != "float32" && word != "float") {
                  Log.WithTime("unknown property datatype :/");
                  return;
               }
               iss >> word;
               if (!(word == "x" || word == "y" || word == "z")) {
                  Log.WithTime("unknown property name :/");
                  //return;
               }
            }
            else if (inFace) {

            }
            else {
               Log.WithTime("property outside of element :/");
               return;
            }
         }
      }

      if (numVertices <= 0) {
         Log.WithTime("header contains no vertex info :/");
         return;
      }

      if (numFaces <= 0) {
         Log.WithTime("header contains no face info :/");
         return;
      }

      // data - vertices

      for (int i = 0; i < numVertices; i++) {
         if (!getline(*stream, line)) {
            Log.WithTime("Failed to read line in vertices :/");
            return;
         }

         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         iss >> word;
         const float x = stof(word);
         iss >> word;
         const float y = stof(word);
         iss >> word;
         const float z = stof(word);
         Point p(x, y, z);
         mesh->ObjectToWorld->ApplyInPlace(p);
         mesh->x.push_back(p.x);
         mesh->y.push_back(p.y);
         mesh->z.push_back(p.z);
         mesh->num_vertices++;
      }

      std::ostringstream str;
      str << "Parsed " << mesh->num_vertices << " vertices.";
      Log.WithTime(str.str());
      str.str("");

      // data - faces

      for (int i = 0; i < numFaces; i++) {
         if (!getline(*stream, line)) {
            Log.WithTime("Failed to read line in faces :/");
            return;
         }

         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         // parse vertex indices

         iss >> word;
         int numVertexIndices = stoui(word);
         if (numVertexIndices != 3) {
            Log.WithTime("Face has too many vertex indices :/");
            return;
         }

         iss >> word;
         // TODO error handling for non-existent face
         const unsigned int v0 = stoui(word);
         iss >> word;
         const unsigned int v1 = stoui(word);
         iss >> word;
         const unsigned int v2 = stoui(word);
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
      }

      str << "Parsed " << mesh->num_faces << " faces.";
      Log.WithTime(str.str());

      mesh->BoundingBox->p0 = min;
      mesh->BoundingBox->p1 = max;
      
      mesh->ExpandFaces();
   }
}
