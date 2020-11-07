//
// Created by Daniel on 27-Dec-19.
//

#ifndef POLY_ABSTRACTMESHPARSER_H
#define POLY_ABSTRACTMESHPARSER_H

#include "abstract_file_parser.h"
#include "../../cpu/shapes/mesh.h"

namespace poly {
   class abstract_mesh_parser : public abstract_file_parser {
   public:
      virtual void parse_file(std::shared_ptr<poly::mesh_geometry> mesh, const std::string &filepath) const = 0;
      bool has_vertex_normals = false;
   };

   class obj_parser : public abstract_mesh_parser {
   public:
      void parse_file(std::shared_ptr<poly::mesh_geometry> mesh, const std::string &filepath) const override;
   };

   class ply_parser : public abstract_mesh_parser {
   public:
      void parse_file(std::shared_ptr<poly::mesh_geometry> mesh, const std::string &filepath) const override;

      
      enum ply_format {
         ascii,
         binary_le,
         binary_be
      };

      enum ply_property_name {
         x,
         y,
         z,
         nx,
         ny,
         nz,
         unknown
      };

      enum ply_element_type {
         vertex,
         face
      };
      
      enum ply_property_type {
         ply_char,
         ply_uchar,
         ply_short,
         ply_ushort,
         ply_int,
         ply_uint,
         ply_float,
         ply_double,
         ply_list,
         ply_unknown
      };
      
      struct ply_property {
         ply_property_name name;
         ply_property_type type;
         ply_property_type list_prefix_type;
         ply_property_type list_elements_type;
      };
      
      struct ply_element {
         std::vector<ply_property> properties;
         ply_element_type type;
         int num_instances;
      };
      
      struct parser_state {
         std::unique_ptr<std::ifstream> stream;
         ply_format data_format;
         unsigned int line_number;
         std::vector<ply_element> elements;
         bool has_vertex_normals;
      };
   
      struct parser_state parse_header(const std::string &filepath) const;
   
      static float read_float(const std::unique_ptr<std::ifstream> &stream, ply_format format) ;
      static int read_int(const std::unique_ptr<std::ifstream> &stream, ply_format format) ;
      static unsigned int read_uint(const std::unique_ptr<std::ifstream> &stream, ply_format format);

      static unsigned char read_uchar(const std::unique_ptr<std::ifstream> &stream);
      
      static ply_property_type get_type_for_token(const std::string& token);
   };
}

#endif //POLY_ABSTRACTMESHPARSER_H
