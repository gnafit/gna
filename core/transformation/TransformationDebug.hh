#pragma once

//#define TRANSFORMATION_DEBUG 1

#ifdef TRANSFORMATION_DEBUG
#define TR_DPRINTF(...) do {                    \
  fprintf(stderr, __VA_ARGS__);                 \
} while (0)
#else
#define TR_DPRINTF(...)
#endif
