//
// Created by Daniel on 07-Apr-18.
//

#ifndef POLY_FILEPARSER_H
#define POLY_FILEPARSER_H

#include <iostream>
#include <string>
#include "../../cpu/runners/AbstractRunner.h"
#include "AbstractFileParser.h"

namespace poly {

   class PBRTArgument {
   public:

      enum PBRTArgumentType {
         pbrt_bool,
         pbrt_float,
         pbrt_int,
         pbrt_normal,
         pbrt_point2,
         pbrt_point3,
         pbrt_rgb,
         pbrt_spectrum,
         pbrt_string,
         pbrt_vector2,
         pbrt_vector3
      } Type;


      PBRTArgument(const PBRTArgumentType type) : Type(type) { 
         switch (type) {
            case pbrt_bool: {
               bool_value = std::make_unique<bool>();
               break;
            }
            case pbrt_float: {
               float_values = std::make_unique<std::vector<float>>();
               break;
            }
            case pbrt_int: {
               int_values = std::make_unique<std::vector<int>>();
               break;
            }
            case pbrt_string: {
               string_value = std::make_unique<std::string>();
               break;
            }
            default: {
               // do nothing until we add further union types
            }
         }
      }
      
      std::string Name;
      std::unique_ptr<bool> bool_value;
      std::unique_ptr<std::vector<float>> float_values;
      std::unique_ptr<std::string> string_value;
      std::unique_ptr<std::vector<int>> int_values;
      
//      union {
//         std::vector<float> float_values;
//         std::string string_value;
//         std::vector<int> int_values;
//      };
      //std::vector<std::string> Values;

      // todo put this into a map
      static PBRTArgumentType get_argument_type(const std::string &value) {
         if (value == "bool")
            return pbrt_bool;
         if (value == "float")
            return pbrt_float;
         if (value == "integer")
            return pbrt_int;
         if (value == "normal" || value == "normal3")
            return pbrt_normal;
         if (value == "point2")
            return pbrt_point2;
         if (value == "point" || value == "point3")
            return pbrt_point3;
         if (value == "rgb" || value == "color")
            return pbrt_rgb;            
         if (value == "spectrum")
            return pbrt_spectrum;
         if (value == "string")
            return pbrt_string;
         if (value == "vector2")
            return pbrt_vector2;
         if (value == "vector" || value == "vector3")
            return pbrt_vector3;
         
         throw std::invalid_argument("Given argument type [" + value + "] is not a known PBRT type");
            
      }
      
      static std::string get_argument_type_string(const PBRTArgumentType type) {
         std::string argument_type;
         switch (type) {
            case pbrt_bool: {
               argument_type = "bool";
               break;
            }
            case pbrt_float: {
               argument_type = "float";
               break;
            }
            case pbrt_int: {
               argument_type = "int";
               break;
            }
            case pbrt_normal: {
               argument_type = "normal";
               break;
            }
            case pbrt_point2: {
               argument_type = "point2";
               break;
            }
            case pbrt_point3: {
               argument_type = "point3";
               break;
            }
            case pbrt_rgb: {
               argument_type = "rgb";
               break;
            }
            case pbrt_spectrum: {
               argument_type = "spectrum";
               break;
            }
            case pbrt_string: {
               argument_type = "string";
               break;
            }
            case pbrt_vector2: {
               argument_type = "vector2";
               break;
            }
            case pbrt_vector3: {
               argument_type = "vector3";
               break;
            }
            default: {
               argument_type = "OOPS";
               break;
            }
         }
         return argument_type;
      }
   };

   class PBRTDirective {
   public:
      std::string Name;
      std::string Identifier;
      std::vector<poly::PBRTArgument> Arguments;
   };

   class PBRTGraphicsState {
   public:
      std::unique_ptr<Material> material;
   };

   class PBRTFileParser : public AbstractFileParser {
   public:

      // constructors
      explicit PBRTFileParser() = default;

      std::unique_ptr<AbstractRunner> ParseFile(const std::string &filepath) noexcept(false);
      
      std::unique_ptr<AbstractRunner> ParseString(const std::string &text) noexcept(false);
      static std::unique_ptr<PBRTDirective> Lex(std::vector<std::string> line);
      static std::unique_ptr<std::vector<std::vector<std::string>>> Scan(const std::unique_ptr<std::istream>& stream);
      std::unique_ptr<AbstractSampler> Sampler;
      Scene* _scene = nullptr;
      std::unique_ptr<AbstractIntegrator> _integrator;
      std::unique_ptr<AbstractFilm> _film;
      std::unique_ptr<AbstractFilter> _filter;
      poly::Bounds _bounds;
   
   private:
      std::unique_ptr<AbstractRunner> Parse(std::unique_ptr<std::vector<std::vector<std::string>>> tokens) noexcept(false);
   };
}
#endif //POLY_FILEPARSER_H
