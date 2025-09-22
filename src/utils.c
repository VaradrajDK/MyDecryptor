#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>

unsigned char* load_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    unsigned char *buf = (unsigned char*)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

bool save_file(const char *path, const unsigned char *buf, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    size_t w = fwrite(buf, 1, len, f);
    fclose(f);
    return w == len;
}

void sha256_hash(const unsigned char *buf, size_t len, unsigned char out[32]) {
    SHA256(buf, len, out);
}

int memeq(const unsigned char *a, const unsigned char *b, size_t len) {
    return memcmp(a,b,len)==0;
}

void init_progress(progress_state *p, const char *resume_path) {
    p->tried = 0;
    if (!resume_path) return;
    FILE *f = fopen(resume_path, "rb");
    if (!f) return;
    fread(&p->tried, sizeof(p->tried), 1, f);
    fclose(f);
}

void save_progress(progress_state *p, const char *resume_path) {
    if (!resume_path) return;
    FILE *f = fopen(resume_path, "wb");
    if (!f) return;
    fwrite(&p->tried, sizeof(p->tried), 1, f);
    fclose(f);
}