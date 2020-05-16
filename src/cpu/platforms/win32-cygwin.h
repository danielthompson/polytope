////
//// Created by Daniel on 15-Dec-19.
////
//
//#ifndef POLYTOPE_WIN32_CYGWIN_H
//#define POLYTOPE_WIN32_CYGWIN_H
//
//#ifdef __CYGWIN__
//
//#include <windows.h>
//#include <dbghelp.h>
//#include <assert.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <imagehlp.h>
//#define BACKTRACE_BT_BUFFER_SIZE 61
//
//char* PROGRAM_NAME = nullptr;
///* Resolve symbol name and source location given the path to the executable
//   and an address */
//int addr2line(char const * const program_name, void const * const addr)
//{
//   char addr2line_cmd[512] = {0};
//   sprintf(addr2line_cmd,"addr2line -f -p -e %.256s %p", program_name, addr);
//   /* This will print a nicely formatted string specifying the
//      function and source line of the address */
//   return system(addr2line_cmd);
//}
//
//void printStack()
//{
//   assert(PROGRAM_NAME != nullptr);
//   unsigned int i;
//   void* stack[BACKTRACE_BT_BUFFER_SIZE];
//   unsigned short frames;
//   SYMBOL_INFO* symbol;
//   HANDLE process;
//
//   process = GetCurrentProcess();
//   SymInitialize(process, nullptr, true);
//   frames = CaptureStackBackTrace(0, BACKTRACE_BT_BUFFER_SIZE, stack, nullptr);
//   for( i = 0; i < frames; i++ )
//   {
//      addr2line(PROGRAM_NAME, stack[i]);
//   }
//}
//
//#endif //__CYGWIN__
//
//#endif //POLYTOPE_WIN32_CYGWIN_H
