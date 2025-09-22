#ifndef KEYGEN_H
#define KEYGEN_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "crypto.h"
#include "cli.h"

typedef struct {
    unsigned char prefix[32];
    unsigned char suffix[32];
    int segment_mode;   // 0 first4, 1 last4, 2 middle4
    uint64_t max;       // maximum keys to try (0 means full 2^32)
} keyspace_spec;

void init_keyspace_from_opts(keyspace_spec *ks, const cli_options *opts);
void construct_key_from_index(uint32_t idx, unsigned char *key, const keyspace_spec *ks, const cipher_spec *spec);

#endif