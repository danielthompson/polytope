//
// Created by Daniel on 27-Dec-19.
//

#include <sstream>
#include <cstring>
#include "PLYParser.h"
#include "../../utilities/Common.h"

namespace Polytope {

   std::unique_ptr<std::ifstream> PLYParser::parse_header(const std::string &filepath, int* num_vertices, int* num_faces, ply_format* format) const {
      std::unique_ptr<std::ifstream> stream = open_ascii_stream(filepath);

      std::string line;

      // ply header
      if (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);
         if (!(iss >> word) || word != "ply") {
            throw std::invalid_argument(filepath + ": Missing PLY header");
         }
      }
      else {
         throw std::invalid_argument(filepath + ": Read error when trying to read PLY header");
      }

      // format
      if (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (!(iss >> word) || word != "format") {
            throw std::invalid_argument(filepath + ": Missing \"format\" header");
         }
         if (!(iss >> word)) {
            throw std::invalid_argument(filepath + ": Unsupported format");
         }
         if (word == "ascii") {
            *format = ply_format::ascii;
         }
         else if (word == "binary_little_endian") {
            *format = ply_format::binary_le;
         }
         else if (word == "binary_big_endian") {
            *format = ply_format::binary_be;
         }
         else {
            throw std::invalid_argument(filepath + ": Read error when reading format type");
         }
         
         if (!(iss >> word) || word != "1.0") {
            throw std::invalid_argument(filepath + ": Unsupported format version");
         }
      }
      else {
         throw std::invalid_argument(filepath + ": Read error when reading format line");
      }

      // rest of header

      bool in_vertex = false;
      bool in_face = false;
      while (getline(*stream, line)) {
         std::string word;
         std::istringstream iss(line, std::istringstream::in);
         iss >> word;
         if (word == "end_header") {
            break;
         }
         else if (word == "element") {
            in_vertex = false;
            in_face = false;
            iss >> word;
            if (word == "vertex") {
               in_vertex = true;
               iss >> word;
               *num_vertices = stoi(word);
            }
            else if (word == "face") {
               in_face = true;
               iss >> word;
               *num_faces = stoi(word);
            }
         }
         else if (word == "property") {
            if (in_vertex) {
               iss >> word;
               if (word != "float32" && word != "float") {
                  throw std::invalid_argument(filepath + ": Unknown property datatype");
               }
               iss >> word;
               if (!(word == "x" || word == "y" || word == "z")) {
                  throw std::invalid_argument(filepath + ": Unknown property name");
               }
            }
            else if (in_face) {

            }
            else {
               throw std::invalid_argument(filepath + ": Property outside of element");
            }
         }
      }

      if (*num_vertices == 0) {
         throw std::invalid_argument(filepath + ": Header contains no vertices");
      }

      if (*num_faces == 0) {
         throw std::invalid_argument(filepath + ": Header contains no faces");
      }

      if (*format == ply_format::binary_le || *format == ply_format::binary_be) {
         std::streampos offset = stream->tellg();
         stream->close();
         stream = open_binary_stream(filepath);
         stream->seekg(offset);
      }
      
      return stream;
   }
   
   float PLYParser::read_float(const std::unique_ptr<std::ifstream> &stream, ply_format format) {
      char buffer[4];
      float value;
      stream->read(buffer, 4);
      if (format == binary_be) {
         std::swap(buffer[0], buffer[3]);
         std::swap(buffer[1], buffer[2]);
      }
      std::memcpy(&value, buffer, 4);
      return value;
   }

   int PLYParser::read_int(const std::unique_ptr<std::ifstream> &stream, ply_format format) {
      char buffer[4];
      int value;
      stream->read(buffer, 4);
      if (format == binary_be) {
         std::swap(buffer[0], buffer[3]);
         std::swap(buffer[1], buffer[2]);
      }
      std::memcpy(&value, buffer, 4);
      return value;
   }

   unsigned char PLYParser::read_uchar(const std::unique_ptr<std::ifstream> &stream) {
      char value;
      stream->read(&value, 1);
      unsigned char unsigned_value = reinterpret_cast<unsigned char&>(value);
      return unsigned_value;
   }

   void PLYParser::ParseFile(TriangleMesh *mesh, const std::string &filepath) const {
      
      int num_vertices = -1;
      int num_faces = -1;
      ply_format format = ply_format::ascii;
      
      std::unique_ptr<std::ifstream> stream = parse_header(filepath, &num_vertices, &num_faces, &format);
      
      // data - vertices

      std::string line;
      
      if (format == ascii) {
         for (int i = 0; i < num_vertices; i++) {
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
      }
      else {
         for (int i = 0; i < num_vertices; i++) {
            const float x = read_float(stream, format);
            const float y = read_float(stream, format);
            const float z = read_float(stream, format);
            const Point p(x, y, z);
            const Point worldPoint = mesh->ObjectToWorld->Apply(p);
            mesh->Vertices.push_back(worldPoint);
         }
      }

      Log.WithTime("Parsed " + std::to_string(mesh->Vertices.size()) + " vertices.");

      // data - faces

      Polytope::Point min(FloatMax, FloatMax, FloatMax), max(-FloatMax, -FloatMax, -FloatMax);
      
      for (int i = 0; i < num_faces; i++) {
         unsigned int v0, v1, v2;
         
         if (format == ascii) {
            if (!getline(*stream, line)) {
               Log.WithTime("Failed to read line in faces :/");
               return;
            }

            std::string word;
            std::istringstream iss(line, std::istringstream::in);

            // parse vertex indices

            iss >> word;
            int numVertexIndices = stoi(word);
            if (numVertexIndices != 3) {
               Log.WithTime("Face has too many vertex indices :/");
               return;
            }

            iss >> word;
            // TODO error handling for non-existent face
            v0 = stoui(word);
            iss >> word;
            v1 = stoui(word);
            iss >> word;
            v2 = stoui(word);
         }
         else {
            const unsigned char num_vertex_indices = read_uchar(stream);
            if (num_vertex_indices != 3) {
               Log.WithTime("Face has too many vertex indices :/");
               return;
            }
            v0 = read_int(stream, format);
            v1 = read_int(stream, format);
            v2 = read_int(stream, format);
         }

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
      }

      Log.WithTime("Parsed " + std::to_string(mesh->Faces.size()) + " faces.");

      mesh->BoundingBox->p0 = min;
      mesh->BoundingBox->p1 = max;
   }

   void PLYParser::ParseFile(TriangleMeshSOA *mesh, const std::string &filepath) const {
      int num_vertices = -1;
      int num_faces = -1;

      ply_format format = ply_format::ascii;
      std::unique_ptr<std::ifstream> stream = parse_header(filepath, &num_vertices, &num_faces, &format);

      // data - vertices

      std::string line;
      
      if (format == ascii) {
         for (int i = 0; i < num_vertices; i++) {
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
      }
      
      else {
         for (int i = 0; i < num_vertices; i++) {
            const float x = read_float(stream, format);
            const float y = read_float(stream, format);
            const float z = read_float(stream, format);
            Point p(x, y, z);
            mesh->ObjectToWorld->ApplyInPlace(p);
            mesh->x.push_back(p.x);
            mesh->y.push_back(p.y);
            mesh->z.push_back(p.z);
            mesh->num_vertices++;
         }
      }

      Log.WithTime("Parsed " + std::to_string(mesh->num_vertices) + " vertices.");
      
      // data - faces

      Polytope::Point min(FloatMax, FloatMax, FloatMax), max(-FloatMax, -FloatMax, -FloatMax);
      
      for (int i = 0; i < num_faces; i++) {
         unsigned int v0, v1, v2;

         if (format == ascii) {
            if (!getline(*stream, line)) {
               Log.WithTime("Failed to read line in faces :/");
               return;
            }
            std::string word;
            std::istringstream iss(line, std::istringstream::in);

            // parse vertex indices

            iss >> word;
            int numVertexIndices = stoi(word);
            if (numVertexIndices != 3) {
               Log.WithTime("Face has too many vertex indices :/");
               return;
            }

            iss >> word;
            // TODO error handling for non-existent face
            v0 = stoui(word);
            iss >> word;
            v1 = stoui(word);
            iss >> word;
            v2 = stoui(word);
         }
         else {
            const unsigned char num_vertex_indices = read_uchar(stream);
            if (num_vertex_indices != 3) {
               Log.WithTime("Face has too many vertex indices :/");
               return;
            }
            v0 = read_int(stream, format);
            v1 = read_int(stream, format);
            v2 = read_int(stream, format);
         }

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

      Log.WithTime("Parsed " + std::to_string(mesh->num_faces) + " faces.");
      
      mesh->BoundingBox->p0 = min;
      mesh->BoundingBox->p1 = max;
      
      mesh->ExpandFaces();
   }
}
