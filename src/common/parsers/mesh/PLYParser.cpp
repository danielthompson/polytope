//
// Created by Daniel on 27-Dec-19.
//

#include <sstream>
#include <cstring>
#include "PLYParser.h"
#include "../../utilities/Common.h"

namespace Polytope {

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
   
   std::unique_ptr<std::ifstream> PLYParser::parse_header(const std::string &filepath, int* num_vertices, int* num_faces, ply_format* format) const {
      std::unique_ptr<std::ifstream> stream = open_ascii_stream(filepath);

      std::string line;

      unsigned int line_number = 0;
      
      // ply header
      if (getline(*stream, line)) {
         line_number++;
         std::string word;
         std::istringstream iss(line, std::istringstream::in);
         if (!(iss >> word) || word != "ply") {
            ERROR("%s:%i Missing PLY header", filepath.c_str(), line_number);
         }
      }
      else {
         ERROR("%s:%i Read error when trying to read PLY header", filepath.c_str(), line_number);
      }

      // format
      if (getline(*stream, line)) {
         line_number++;
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         if (!(iss >> word) || word != "format") {
            ERROR("%s:%i Missing \"format\" header", filepath.c_str(), line_number);
         }
         if (!(iss >> word)) {
            ERROR("%s:%i Read error when reading format type", filepath.c_str(), line_number);
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
            ERROR("%s:%i Unsupported format [%s]", filepath.c_str(), line_number, word.c_str());
         }
         
         if (!(iss >> word)) {
            ERROR("%s:%i Read error when reading format version", filepath.c_str(), line_number);
         }
         if (word != "1.0") {
            ERROR("%s:%i Unsupported format version [%s]", filepath.c_str(), line_number, word.c_str());
         }
      }
      else {
         ERROR("%s:%i Read error when reading format line", filepath.c_str(), line_number);
      }

      // rest of header

      bool has_x, has_y, has_z;
      
      bool in_vertex = false;
      bool in_face = false;
      while (getline(*stream, line)) {
         line_number++;
         std::string word;
         std::istringstream iss(line, std::istringstream::in);
         iss >> word;
         if (word == "end_header") {
            if (!has_x) {
               ERROR("%s:%i Header missing element x property", filepath.c_str(), line_number);
            }
            if (!has_y) {
               ERROR("%s:%i Header missing element y property", filepath.c_str(), line_number);
            }
            if (!has_z) {
               ERROR("%s:%i Header missing element z property", filepath.c_str(), line_number);
            }
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
                  WARNING("%s:%i Unknown property type [%s]", filepath.c_str(), line_number, word.c_str());
                  continue;
               }
               iss >> word;
               if (word == "x") {
                  has_x = true;
                  continue;
               }
               else if (word == "y") {
                  has_y = true;
                  continue;
               }
               else if (word == "z") {
                  has_z = true;
                  continue;
               }
               else {
                  WARNING("%s:%i Ignoring unknown property name [%s]", filepath.c_str(), line_number, word.c_str());
               }
            }
            else if (in_face) {

            }
            else {
               ERROR("%s:%i Property outside of element", filepath.c_str(), line_number);
            }
         }
      }
      
      if (*num_vertices == 0) {
         ERROR("%s:%i Header contains no vertices", filepath.c_str(), line_number);
      }

      if (*num_faces == 0) {
         ERROR("%s:%i Header contains no faces", filepath.c_str(), line_number);
      }

      if (*format == ply_format::binary_le || *format == ply_format::binary_be) {
         std::streampos offset = stream->tellg();
         stream->close();
         stream = open_binary_stream(filepath);
         stream->seekg(offset);
      }
      
      return stream;
   }

   void PLYParser::ParseFile(AbstractMesh *mesh, const std::string &filepath) const {
      int num_vertices = -1;
      int num_faces = -1;

      ply_format format = ply_format::ascii;
      std::unique_ptr<std::ifstream> stream = parse_header(filepath, &num_vertices, &num_faces, &format);

      // data - vertices

      std::string line;
      
      if (format == ascii) {
         std::string word;
         Point p;
         for (int i = 0; i < num_vertices; i++) {
            if (!getline(*stream, line)) {
               Log.WithTime("Failed to read line in vertices :/");
               return;
            }

            word.clear();
            std::istringstream iss(line, std::istringstream::in);
            iss >> word;
            p.x = stof(word);
            iss >> word;
            p.y = stof(word);
            iss >> word;
            p.z = stof(word);
            mesh->add_vertex(p);
         }
      }
      
      else {
         for (int i = 0; i < num_vertices; i++) {
            const float x = read_float(stream, format);
            const float y = read_float(stream, format);
            const float z = read_float(stream, format);
            Point p(x, y, z);
            mesh->add_vertex(p.x, p.y, p.z);
         }
      }

      Log.WithTime("Parsed " + std::to_string(mesh->num_vertices) + " vertices.");
      
      // data - faces

      if (format == ascii) {
         for (int i = 0; i < num_faces; i++) {
            unsigned int v0, v1, v2;
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
            mesh->add_packed_face(v0, v1, v2);
         }
      }
      else {
         for (int i = 0; i < num_faces; i++) {
            unsigned int v0, v1, v2;
            const unsigned char num_vertex_indices = read_uchar(stream);
            if (num_vertex_indices != 3) {
               Log.WithTime("Face has too many vertex indices :/");
               return;
            }
            v0 = read_int(stream, format);
            v1 = read_int(stream, format);
            v2 = read_int(stream, format);
            mesh->add_packed_face(v0, v1, v2);
         }
      }

      mesh->unpack_faces();
      Log.WithTime("Parsed " + std::to_string(mesh->num_faces) + " faces.");
   }
}
