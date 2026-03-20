// Stubs for POSIX functions missing in Zephyr's picolibc
// Force-included when compiling GTest for bare-metal
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
static inline int isatty(int fd) { (void)fd; return 0; }
#ifdef __cplusplus
}
#endif
