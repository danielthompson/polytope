//
// Created by Daniel on 27-Dec-19.
//

#ifndef POLY_PLYPARSER_H
#define POLY_PLYPARSER_H

#include "../AbstractFileParser.h"
#include "AbstractMeshParser.h"

namespace poly {
   class PLYParser : public AbstractMeshParser {
   public:
      void ParseFile(Mesh *mesh, const std::string &filepath) const override;

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

#endif //POLY_PLYPARSER_H
