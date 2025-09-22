#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

unsigned char* load_file(const char *path, size_t *out_len);
bool save_file(const char *path, const unsigned char *buf, size_t len);

void sha256_hash(const unsigned char *buf, size_t len, unsigned char out[32]);
int memeq(const unsigned char *a, const unsigned char *b, size_t len);

typedef struct {
    uint64_t tried;
} progress_state;

void init_progress(progress_state *p, const char *resume_path);
void save_progress(progress_state *p, const char *resume_path);

#endif