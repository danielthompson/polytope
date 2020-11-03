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

   private:
      enum ply_format {
         ascii,
         binary_le,
         binary_be
      };

      struct parser_state {
         std::unique_ptr<std::ifstream> stream;
         unsigned int line_number;
         bool has_vertex_normals;
      };

      struct parser_state parse_header(
            const std::string &filepath,
            int *num_faces,
            int *num_vertices,
            ply_format* format) const;

      // TODO move these up to AMP
      static float read_float(const std::unique_ptr<std::ifstream> &stream, ply_format format) ;
      static int read_int(const std::unique_ptr<std::ifstream> &stream, ply_format format) ;
      static unsigned int read_uint(const std::unique_ptr<std::ifstream> &stream, ply_format format);

      static unsigned char read_uchar(const std::unique_ptr<std::ifstream> &stream);
   };
}

#endif //POLY_ABSTRACTMESHPARSER_H
