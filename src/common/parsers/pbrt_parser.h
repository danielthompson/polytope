//
// Created by Daniel on 07-Apr-18.
//

#ifndef POLY_FILEPARSER_H
#define POLY_FILEPARSER_H

#include <iostream>
#include <string>
#include "../../cpu/runners/AbstractRunner.h"
#include "abstract_file_parser.h"

namespace poly {

   class pbrt_argument {
   public:

      enum pbrt_argument_type {
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


      pbrt_argument(const pbrt_argument_type type) : Type(type) { 
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
      static pbrt_argument_type get_argument_type(const std::string &value) {
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
      
      static std::string get_argument_type_string(const pbrt_argument_type type) {
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

   class pbrt_directive {
   public:
      std::string name;
      std::string identifier;
      std::vector<poly::pbrt_argument> arguments;
   };

   class pbrt_graphics_state {
   public:
      std::unique_ptr<Material> material;
   };

   class pbrt_parser : public abstract_file_parser {
   public:

      // constructors
      explicit pbrt_parser() = default;

      std::unique_ptr<AbstractRunner> parse_file(const std::string &filepath) noexcept(false);
      
      std::unique_ptr<AbstractRunner> parse_string(const std::string &text) noexcept(false);
      static std::unique_ptr<pbrt_directive> lex(std::vector<std::string> line);
      static std::unique_ptr<std::vector<std::vector<std::string>>> scan(const std::unique_ptr<std::istream>& stream);
      std::unique_ptr<AbstractSampler> sampler;
      Scene* scene = nullptr;
      std::unique_ptr<AbstractIntegrator> integrator;
      std::unique_ptr<AbstractFilm> film;
      std::unique_ptr<AbstractFilter> filter;
      poly::Bounds bounds;
   
   private:
      std::unique_ptr<AbstractRunner> parse(std::unique_ptr<std::vector<std::vector<std::string>>> tokens) noexcept(false);
   };
}
#endif //POLY_FILEPARSER_H
