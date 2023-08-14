//
// Created by daniel on 10/1/21.
//

#ifndef POLYTOPE_GL_RECORDER_H
#define POLYTOPE_GL_RECORDER_H

#include <cstdio>
#include <string>

namespace poly {
   class gl_recorder {
   public:

      FILE* output;
      int width, height;
      unsigned char* pixels;
      unsigned int framebuffer_size;
      bool stopped = true;
      
      gl_recorder(const int width, const int height) 
      : width(width), height(height) {
         framebuffer_size = width * height * 3;
         pixels = new unsigned char[framebuffer_size];
      }
      
      void start() {
         if (stopped) {
            std::string command = "ffmpeg -y -f rawvideo -s " + std::to_string(width) + "x" + std::to_string(height)
                                  + " -pix_fmt rgb24 -r 60 -i - -vf vflip -an -b:v 2000k test4.mp4";
            output = popen(command.c_str(), "w");
            stopped = false;
            LOG_INFO("Started recording.");
         }
         else {
            LOG_DEBUG("Already started.");
         }
      }
      
      void capture_frame() {
         if (!stopped) {
            gl::glReadPixels(0, 0, width, height, gl::GL_RGB, gl::GL_UNSIGNED_BYTE, pixels);
            fwrite(pixels, framebuffer_size, 1, output);
         }
      }
      
      void stop() {
         if (stopped) {
            LOG_DEBUG("Already stopped.");
         } else {
            pclose(output);
            stopped = true;
            LOG_INFO("Stopped recording.");
         }
      }
      
      ~gl_recorder() {
         stop();
         delete pixels;
      }
   };
}


#endif //POLYTOPE_GL_RECORDER_H

