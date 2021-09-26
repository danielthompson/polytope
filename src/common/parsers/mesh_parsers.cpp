//
// Created by Daniel on 27-Dec-19.
//

#include <sstream>
#include <cstring>
#include "mesh_parsers.h"
#include "../utilities/Common.h"
#include "../../cpu/shapes/mesh.h"

namespace poly {

   
   
   float ply_parser::read_float(const std::unique_ptr<std::ifstream> &stream, ply_format format) {
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

   int ply_parser::read_int(const std::unique_ptr<std::ifstream> &stream, ply_format format) {
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

   unsigned char ply_parser::read_uchar(const std::unique_ptr<std::ifstream> &stream) {
      char value;
      stream->read(&value, 1);
      unsigned char unsigned_value = reinterpret_cast<unsigned char&>(value);
      return unsigned_value;
   }

   ply_parser::ply_property_type ply_parser::get_type_for_token(const std::string& token) {
      
      // 1 byte
      
      if (token == "char" || token == "int8" || token == "int_8") {
         return ply_char;
      }
      else if (token == "uchar" || token == "u_char" || token == "uint8" || token == "u_int8" || token == "uint_8" || token == "u_int_8") {
         return ply_uchar;
      }

      // 2 bytes

      else if (token == "short" || token == "int16" || token == "int_16") {
         return ply_short;
      }
      else if (token == "ushort" || token == "u_short" || token == "uint16" || token == "u_int16" || token == "uint_16" || token == "u_int_16") {
         return ply_ushort;
      }

      // 4 bytes

      else if (token == "int" || token == "int32" || token == "int_32") {
         return ply_int;
      }
      else if (token == "uint" || token == "u_int" || token == "uint32" || token == "u_int32" || token == "uint_32" || token == "u_int_32") {
         return ply_uint;
      }
      else if (token == "float" || token == "float32" || token == "float_32") {
         return ply_float;
      }

      // 8 bytes

      else if (token == "double" || token == "float64" || token == "float_64") {
         return ply_double;
      }

      // list

      else if (token == "list") {
         return ply_list;
      }
      
      return ply_unknown;
   }
   
   struct ply_parser::parser_state ply_parser::parse_header(const std::string &filepath) const {

      struct parser_state state;
      state.stream = abstract_file_parser::open_ascii_stream(filepath);

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
            state.data_format = ply_format::ascii;
         }
         else if (word == "binary_little_endian") {
            state.data_format = ply_format::binary_le;
         }
         else if (word == "binary_big_endian") {
            state.data_format = ply_format::binary_be;
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
      bool has_x = false, has_y = false, has_z = false;
      bool has_nx = false, has_ny = false, has_nz = false;
      bool has_u = false, has_v = false;
      
      state.elements = std::vector<ply_element>();
      while (getline(*state.stream, line)) {
         line_number++;
         std::string word;
         std::istringstream iss(line, std::istringstream::in);
         iss >> word;
         if (word == "element") {
            ply_element element;
            std::string element_name;
            
            // element name
            iss >> element_name;
            if (element_name.empty()) {
               ERROR("%s:%i Element name may not be empty", filepath.c_str(), line_number, element_name.c_str());
            }
            if (element_name == "vertex") {
               element.type = ply_element_type::vertex;
            }
            else if (element_name == "face") {
               element.type = ply_element_type::face;
            }
            else {
               Log.warning("%s:%i Element name [%s] not recognized; will ignore data for it...", filepath.c_str(), line_number, element_name.c_str());
            }
            
            // number of instances
            iss >> word;
            try {
               element.num_instances = stoi(word);
            }
            catch (...) {
               ERROR("%s:%i Failed to parse [%s] as an int", filepath.c_str(), line_number, word.c_str());
            }
            if (element.num_instances == 0) {
               Log.warning("%s:%i Element [%s] declares 0 instances", filepath.c_str(), line_number, element_name.c_str());
            }
            else if (element.num_instances < 0) {
               ERROR("%s:%i Element [%s] must declare a non-negative number of instances, but found [%i]", filepath.c_str(), line_number, element_name.c_str(), element.num_instances);
            }

            state.elements.push_back(element);
            continue;
         }
         else if (word == "property") {
            if (state.elements.empty()) {
               ERROR("%s:%i Found property with no valid preceding element", filepath.c_str(), line_number);
            }

            ply_element& current_element = state.elements[state.elements.size() - 1];
            ply_property current_property;
            
            // get type of property
            iss >> word;
            current_property.type = get_type_for_token(word);
            if (current_property.type == ply_unknown) {
               ERROR("%s:%i Unrecognized property type [%s]", filepath.c_str(), line_number, word.c_str());
            }
            
            switch (current_element.type) {
               case ply_element_type::vertex: {
                  iss >> word;
                  if (word == "x") {
                     current_property.name = ply_property_name::x;
                     has_x = true;
                     break;
                  }
                  else if (word == "y") {
                     current_property.name = ply_property_name::y;
                     has_y = true;
                     break;
                  }
                  else if (word == "z") {
                     current_property.name = ply_property_name::z;
                     has_z = true;
                     break;
                  }
                  else if (word == "nx") {
                     current_property.name = ply_property_name::nx;
                     has_nx = true;
                     break;
                  }
                  else if (word == "ny") {
                     current_property.name = ply_property_name::ny;
                     has_ny = true;
                     break;
                  }
                  else if (word == "nz") {
                     current_property.name = ply_property_name::nz;
                     has_nz = true;
                     break;
                  }
                  else if (word == "u" || word == "s") {
                     current_property.name = ply_property_name::u;
                     has_u = true;
                     break;
                  }
                  else if (word == "v" || word == "t") {
                     current_property.name = ply_property_name::v;
                     has_v = true;
                     break;
                  }
                  else {
                     Log.warning("%s:%i Ignoring unknown property name [%s] for vertex element", filepath.c_str(), line_number, word.c_str());
                     current_property.name = ply_property_name::unknown;
                  }
                  break;
               }
               case ply_element_type::face: {
                  if (current_property.type != ply_property_type::ply_list) {
                     ERROR("%s:%i Face property must have type list (for now) but found type [%s]", filepath.c_str(), line_number, word.c_str());
                  }

                  iss >> word;
                  current_property.list_prefix_type = get_type_for_token(word);
                  if (current_property.list_prefix_type == ply_property_type::ply_unknown) {
                     ERROR("%s:%i Face property list has unknown prefix type [%s]", filepath.c_str(), line_number, word.c_str());
                  }

                  iss >> word;
                  current_property.list_elements_type = get_type_for_token(word);
                  if (current_property.list_elements_type == ply_property_type::ply_unknown) {
                     ERROR("%s:%i Face property list has unknown elements type [%s]", filepath.c_str(), line_number, word.c_str());
                  }

                  iss >> word;
                  if (word != "vertex_indices" && word != "vertex_index" && word != "vertices" && word != "vertex") {
                     ERROR("%s:%i Face property list has unknown name [%s], expecting [vertex_indices]", filepath.c_str(), line_number, word.c_str());
                  }
                  break;
               }
               default: {
                  ERROR("%s:%i Property outside of element", filepath.c_str(), line_number);
               }
            }
            
            current_element.properties.push_back(current_property);
         }
         else if (word == "end_header") {
            if (!has_x) {
               ERROR("%s:%i Header missing element vertex x property", filepath.c_str(), line_number);
            }
            if (!has_y) {
               ERROR("%s:%i Header missing element vertex y property", filepath.c_str(), line_number);
            }
            if (!has_z) {
               ERROR("%s:%i Header missing element vertex z property", filepath.c_str(), line_number);
            }
            
            int num_normals = has_nx + has_ny + has_nz;
            
            // must have all 3 vertex normals or none
            if (num_normals != 3 && num_normals != 0) {
               ERROR("%s:%i Header has %i normal directions per vertex; must have all (3) or none (0)", filepath.c_str(), line_number, num_normals);
            }
            
            state.has_vertex_normals = num_normals;
            
            int num_uv = has_u + has_v;
            
            // must have both u and v, or neither
            if (num_uv != 0 && num_uv != 2) {
               ERROR("%s:%i Header has %i uv per vertex; must have all (2) or none (0)", filepath.c_str(), line_number, num_uv);
            }
            
            state.has_vertex_uv = true;
            
            break;
         }
      }

      if (state.data_format == ply_format::binary_le || state.data_format == ply_format::binary_be) {
         std::streampos offset = state.stream->tellg();
         state.stream->close();
         state.stream = abstract_file_parser::open_binary_stream(filepath);
         state.stream->seekg(offset);
      }
      
      return state;
   }

   void ply_parser::parse_file(std::shared_ptr<poly::mesh_geometry> mesh, const std::string &filepath) const {
      struct parser_state state = parse_header(filepath);
      
      mesh->has_vertex_normals = state.has_vertex_normals;
      mesh->has_vertex_uvs = state.has_vertex_uv;
      
      for (const auto& element : state.elements) {
         switch (element.type) {
            case ply_element_type::vertex: {
               std::string line;
               
               if (state.data_format == ply_format::ascii) {
                  std::string word;
                  poly::point p;
                  normal n;
                  float u, v;
                  for (int i = 0; i < element.num_instances; i++) {
                     if (!getline(*state.stream, line)) {
                        ERROR("%s:%i Error reading vertex %i", filepath.c_str(), state.line_number, i);
                     }
                     word.clear();
                     std::istringstream iss(line, std::istringstream::in);
                     for (const auto& property : element.properties) {
                        iss >> word;
                        // TODO investigate whether we need to ever parse this as anything other than float
                        float parsed_value = stof(word);
                        switch (property.name) {
                           case ply_property_name::x: {
                              p.x = parsed_value;
                              break;
                           }
                           case ply_property_name::y: {
                              p.y = parsed_value;
                              break;
                           }
                           case ply_property_name::z: {
                              p.z = parsed_value;
                              break;
                           }
                           case ply_property_name::nx: {
                              n.x = parsed_value;
                              break;
                           }
                           case ply_property_name::ny: {
                              n.y = parsed_value;
                              break;
                           }
                           case ply_property_name::nz: {
                              n.z = parsed_value;
                              break;
                           }
                           case ply_property_name::u: {
                              u = parsed_value;
                              break;
                           }
                           case ply_property_name::v: {
                              v = parsed_value;
                              break;
                           }
                           case ply_property_name::unknown: {
                              // do nothing, since we already warned about this above
                              break;
                           }
                        }
                     }
                     if (state.has_vertex_normals) {
                        if (state.has_vertex_uv) {
                           mesh->add_vertex(p, n, u, v);
                        }
                        else {
                           mesh->add_vertex(p, n);
                        }
                     }
                     else {
                        if (state.has_vertex_uv) {
                           mesh->add_vertex(p, u, v);
                        }
                        else {
                           mesh->add_vertex(p);   
                        }
                     }
                  }
               }
               else {
                  for (int i = 0; i < element.num_instances; i++) {
                     point p;
                     normal n;
                     float u, v;
                     for (const auto& property : element.properties) {
                        float parsed_value = read_float(state.stream, state.data_format);
                        switch (property.name) {
                           case ply_property_name::x: {
                              p.x = parsed_value;
                              break;
                           }
                           case ply_property_name::y: {
                              p.y = parsed_value;
                              break;
                           }
                           case ply_property_name::z: {
                              p.z = parsed_value;
                              break;
                           }
                           case ply_property_name::nx: {
                              n.x = parsed_value;
                              break;
                           }
                           case ply_property_name::ny: {
                              n.y = parsed_value;
                              break;
                           }
                           case ply_property_name::nz: {
                              n.z = parsed_value;
                              break;
                           }
                           case ply_property_name::u: {
                              u = parsed_value;
                              break;
                           }
                           case ply_property_name::v: {
                              v = parsed_value;
                              break;
                           }
                           case ply_property_name::unknown: {
                              // do nothing, since we already warned about this above
                              break;
                           }
                        }
                     }

                     if (state.has_vertex_normals) {
                        if (state.has_vertex_uv) {
                           mesh->add_vertex(p, n, u, v);
                        }
                        else {
                           mesh->add_vertex(p, n);
                        }
                     }
                     else {
                        if (state.has_vertex_uv) {
                           mesh->add_vertex(p, u, v);
                        }
                        mesh->add_vertex(p);
                     }
                  }
               }

               Log.debug("Parsed " + add_commas(mesh->num_vertices_packed) + " vertices.");
               
               continue;
            }
            case ply_element_type::face: {

               // data - faces

               if (state.data_format == ascii) {
                  std::string line;
                  for (int i = 0; i < element.num_instances; i++) {
                     // TODO respect declared list member types
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
                        ERROR("%s:%i Face instance has wrong number of vertex indices (expected 3, found %i)", filepath.c_str(), state.line_number, num_vertex_indices);
                     }

                     iss >> word;
                     // TODO error handling for non-existent face
                     v0 = abstract_file_parser::stoui(word);
                     iss >> word;
                     v1 = abstract_file_parser::stoui(word);
                     iss >> word;
                     v2 = abstract_file_parser::stoui(word);
                     mesh->add_packed_face(v0, v1, v2);
                  }
               }
               else {
                  for (int i = 0; i < element.num_instances; i++) {
                     unsigned int v0, v1, v2;
                     const unsigned char num_vertex_indices = read_uchar(state.stream);
                     if (num_vertex_indices != 3) {
                        ERROR("%s:%i Face has wrong number of vertex indices (expected 3, found %i)", filepath.c_str(), state.line_number, num_vertex_indices);
                     }
                     v0 = read_int(state.stream, state.data_format);
                     v1 = read_int(state.stream, state.data_format);
                     v2 = read_int(state.stream, state.data_format);
                     mesh->add_packed_face(v0, v1, v2);
                  }
               }

               Log.debug("Parsed " + add_commas(mesh->num_faces) + " faces.");
               
               continue;
            }
            default: {
               ERROR("%s:%i Unknown element type", filepath.c_str(), state.line_number);
            }
         }
      }

      mesh->unpack_faces();
   }

   void obj_parser::parse_file(std::shared_ptr<poly::mesh_geometry> mesh, const std::string &filepath) const {

      std::unique_ptr<std::istream> stream = abstract_file_parser::open_ascii_stream(filepath);
      std::string line;
      point p;
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
                  const unsigned int v0 = abstract_file_parser::stoui(word) - 1;
                  iss >> word;
                  const unsigned int v1 = abstract_file_parser::stoui(word) - 1;
                  iss >> word;
                  const unsigned int v2 = abstract_file_parser::stoui(word) - 1;
                  mesh->add_packed_face(v0, v1, v2);
                  continue;
               }
               default: {
                  Log.warning("OBJ Parser: Ignoring line with unimplemented first char [%s]", firstChar);
               }
            }
         }
      }

      mesh->unpack_faces();
   }
}
