//
// Created by Daniel on 27-Dec-19.
//

#ifndef POLYTOPE_PLYPARSER_H
#define POLYTOPE_PLYPARSER_H

#include "../AbstractFileParser.h"
#include "AbstractMeshParser.h"


namespace Polytope {
   class PLYParser : public AbstractMeshParser {
   public:
      void ParseFile(AbstractMesh *mesh, const std::string &filepath) const override;

   private:
      enum ply_format {
         ascii,
         binary_le,
         binary_be
      };
      std::unique_ptr<std::ifstream> parse_header(const std::string &filepath, int *num_faces, int *num_vertices, ply_format* format) const;

      // TODO move these up to AMP
      static float read_float(const std::unique_ptr<std::ifstream> &stream, ply_format format) ;
      static int read_int(const std::unique_ptr<std::ifstream> &stream, ply_format format) ;
      static unsigned int read_uint(const std::unique_ptr<std::ifstream> &stream, ply_format format);

      static unsigned char read_uchar(const std::unique_ptr<std::ifstream> &stream);
   };
}

#endif //POLYTOPE_PLYPARSER_H
