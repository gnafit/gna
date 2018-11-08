#pragma once

//#define DEBUG_PARAMETERS

#ifdef DEBUG_PARAMETERS
#define DPRINTF(...) do {                                               \
    const changeable &c = *static_cast<const changeable*>(this);        \
    fprintf(stderr, "[%p/%s] ", c.rawdata(), c.name());                 \
    fprintf(stderr, __VA_ARGS__);                                       \
    fprintf(stderr, "\n");                                              \
  } while (0)

#define DPRINTFS(...) do {                                              \
    fprintf(stderr, __VA_ARGS__);                                       \
    fprintf(stderr, "\n");                                              \
  } while (0)
#else
#define DPRINTF(...)
#define DPRINTFS(...)
#endif

