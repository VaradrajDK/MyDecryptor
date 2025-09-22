#ifndef GPU_H
#define GPU_H

#include <stddef.h>
#include <stdint.h>
#include "crypto.h"
#include "keygen.h"
#include "utils.h"

int gpu_search(const unsigned char *enc, size_t enc_len, const unsigned char orig_hash[32],
               const keyspace_spec *ks, const cipher_spec *spec, const unsigned char zip_header[4],
               progress_state *prog, found_key *out);

int cpu_search(const unsigned char *enc, size_t enc_len, const unsigned char orig_hash[32],
               const keyspace_spec *ks, const cipher_spec *spec, const unsigned char zip_header[4],
               progress_state *prog, found_key *out, unsigned char *dec_buf);

#endif