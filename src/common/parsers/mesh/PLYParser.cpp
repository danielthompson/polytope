//
// Created by Daniel on 27-Dec-19.
//

#include <sstream>
#include <cstring>
#include "PLYParser.h"
#include "../../utilities/Common.h"
#include "../../../cpu/shapes/mesh.h"

namespace poly {

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

   struct PLYParser::parser_state PLYParser::parse_header(const std::string &filepath, int* num_vertices, int* num_faces, ply_format* format) const {

      struct parser_state state;
      state.stream = AbstractFileParser::open_ascii_stream(filepath);

      std::string line;

      unsigned int line_number = 0;
      
      // ply header
      if (getline(*state.stream, line)) {
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
      if (getline(*state.stream, line)) {
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
      bool has_nx, has_ny, has_nz;
      has_nx = 0;
      has_ny = 0;
      has_nz = 0;
      
      bool in_vertex = false;
      bool in_face = false;
      while (getline(*state.stream, line)) {
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
            // must have all 3 vertex normals or none
            if (!((has_nx && has_ny && has_nz) || (!has_nx && !has_ny && !has_nz))) {
               int normals = has_nx + has_ny + has_nz;
               ERROR("%s:%i Header has %i normal directions per vertex; must have all (3) or none (0)", filepath.c_str(), line_number, normals);
            }
            
            if (has_nx && has_ny && has_nz) {
               state.has_vertex_normals = true;
            }
            else {
               state.has_vertex_normals = false;
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
                  Log.warning("%s:%i Ignoring unknown property type [%s]", filepath.c_str(), line_number, word.c_str());
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
               else if (word == "nx") {
                  has_nx = true;
                  continue;
               }
               else if (word == "ny") {
                  has_ny = true;
                  continue;
               }
               else if (word == "nz") {
                  has_nz = true;
                  continue;
               }
               else {
                  Log.warning("%s:%i Ignoring unknown property name [%s]", filepath.c_str(), line_number, word.c_str());
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
         std::streampos offset = state.stream->tellg();
         state.stream->close();
         state.stream = AbstractFileParser::open_binary_stream(filepath);
         state.stream->seekg(offset);
      }
      
      return state;
   }

   void PLYParser::ParseFile(Mesh *mesh, const std::string &filepath) const {
      int num_vertices = -1;
      int num_faces = -1;

      ply_format format = ply_format::ascii;
      struct parser_state state = parse_header(filepath, &num_vertices, &num_faces, &format);
      
      mesh->has_vertex_normals = state.has_vertex_normals;
      
      // data - vertices

      std::string line;
      
      if (format == ascii) {
         std::string word;
         Point v;
         for (int i = 0; i < num_vertices; i++) {
            if (!getline(*state.stream, line)) {
               ERROR("%s:%i Error reading vertex %i", filepath.c_str(), state.line_number, i);
            }

            // TODO fix such that property order is not hardcoded
            
            word.clear();
            std::istringstream iss(line, std::istringstream::in);
            iss >> word;
            v.x = stof(word);
            iss >> word;
            v.y = stof(word);
            iss >> word;
            v.z = stof(word);
            
            if (state.has_vertex_normals) {
               Normal n;
               iss >> word;
               n.x = stof(word);
               iss >> word;
               n.y = stof(word);
               iss >> word;
               n.z = stof(word);
               mesh->add_vertex(v, n);
            }
            else {
               mesh->add_vertex(v);
            }
         }
      }
      
      else {
         for (int i = 0; i < num_vertices; i++) {
            const float x = read_float(state.stream, format);
            const float y = read_float(state.stream, format);
            const float z = read_float(state.stream, format);
            Point p(x, y, z);
            
            if (state.has_vertex_normals) {
               const float nx = read_float(state.stream, format);
               const float ny = read_float(state.stream, format);
               const float nz = read_float(state.stream, format);
               Normal n(nx, ny, nz);
               mesh->add_vertex(p, n);
            }
            else {
               mesh->add_vertex(p);   
            }
         }
      }

      Log.debug("Parsed " + add_commas(mesh->num_vertices_packed) + " vertices.");
      
      // data - faces

      if (format == ascii) {
         for (int i = 0; i < num_faces; i++) {
            unsigned int v0, v1, v2;
            if (!getline(*state.stream, line)) {
               ERROR("%s:%i Failed to read face line", filepath.c_str(), state.line_number);
            }
            std::string word;
            std::istringstream iss(line, std::istringstream::in);

            // parse vertex indices

            iss >> word;
            int num_vertex_indices = stoi(word);
            if (num_vertex_indices != 3) {
               ERROR("%s:%i Face has wrong number of vertex indices (expected 3, found %i)", filepath.c_str(), state.line_number, num_vertex_indices);
            }

            iss >> word;
            // TODO error handling for non-existent face
            v0 = AbstractFileParser::stoui(word);
            iss >> word;
            v1 = AbstractFileParser::stoui(word);
            iss >> word;
            v2 = AbstractFileParser::stoui(word);
            mesh->add_packed_face(v0, v1, v2);
         }
      }
      else {
         for (int i = 0; i < num_faces; i++) {
            unsigned int v0, v1, v2;
            const unsigned char num_vertex_indices = read_uchar(state.stream);
            if (num_vertex_indices != 3) {
               ERROR("%s:%i Face has wrong number of vertex indices (expected 3, found %i)", filepath.c_str(), state.line_number, num_vertex_indices);
               return;
            }
            v0 = read_int(state.stream, format);
            v1 = read_int(state.stream, format);
            v2 = read_int(state.stream, format);
            mesh->add_packed_face(v0, v1, v2);
         }
      }

      mesh->unpack_faces();
      Log.debug("Parsed " + add_commas(mesh->num_faces) + " faces.");
   }
}
