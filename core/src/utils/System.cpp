/**
 * @file System.cpp
 * @brief System-level utilities (process memory measurement).
 */
#include "utils/System.hpp"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace ppforest2::sys {
  long get_peak_rss_bytes() {
    #ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
      return static_cast<long>(pmc.PeakWorkingSetSize);
    }

    return -1;

    #else
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      #ifdef __APPLE__
      return usage.ru_maxrss;          // macOS: already in bytes

      #else
      return usage.ru_maxrss * 1024L;  // Linux: reported in KB

      #endif
    }

    return -1;

    #endif // ifdef _WIN32
  }
}
