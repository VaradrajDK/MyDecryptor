#include "keygen.h"
#include <string.h>
#include <stdio.h>

static int hex2bytes(const char *hex, unsigned char *out, size_t out_len) {
    size_t n = strlen(hex);
    if (n != out_len*2) return 0;
    for (size_t i=0;i<out_len;i++) {
        unsigned int b;
        sscanf(hex + 2*i, "%2x", &b);
        out[i] = (unsigned char)b;
    }
    return 1;
}

void init_keyspace_from_opts(keyspace_spec *ks, const cli_options *opts) {
    memset(ks, 0, sizeof(*ks));
    if (opts->segment && strcmp(opts->segment,"first4")==0) ks->segment_mode = 0;
    else if (opts->segment && strcmp(opts->segment,"last4")==0) ks->segment_mode = 1;
    else ks->segment_mode = 2; // middle4 default

    ks->max = (opts->max_keys==0) ? (uint64_t)4294967296ULL : opts->max_keys;

    memset(ks->prefix, 0, 32);
    memset(ks->suffix, 0, 32);
    if (opts->prefix_hex[0]) hex2bytes(opts->prefix_hex, ks->prefix, 32);
    if (opts->suffix_hex[0]) hex2bytes(opts->suffix_hex, ks->suffix, 32);
}

void construct_key_from_index(uint32_t idx, unsigned char *key, const keyspace_spec *ks, const cipher_spec *spec) {
    memcpy(key, ks->prefix, spec->key_len);

    unsigned char segment[4] = {
        (unsigned char)((idx >> 24) & 0xFF),
        (unsigned char)((idx >> 16) & 0xFF),
        (unsigned char)((idx >> 8) & 0xFF),
        (unsigned char)(idx & 0xFF)
    };

    if (ks->segment_mode == 0) {
        memcpy(key, segment, 4);
    } else if (ks->segment_mode == 1) {
        memcpy(key + spec->key_len - 4, segment, 4);
    } else {
        int mid = spec->key_len/2 - 2;
        memcpy(key + mid, segment, 4);
    }

    for (int i=0;i<spec->key_len;i++) {
        if (ks->suffix[i] != 0) key[i] = ks->suffix[i];
    }
}