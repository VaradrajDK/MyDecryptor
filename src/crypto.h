#ifndef CRYPTO_H
#define CRYPTO_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int is_aes;        // 1 for AES, 0 for DES
    int key_len;       // bytes (32 for AES-256, 8 for DES key material)
    int block_len;     // bytes (16 for AES, 8 for DES)
    int use_cbc;       // 1 if CBC, else ECB
    int iv_from_file;  // IV taken from encrypted file first block
    unsigned char iv[16];
} cipher_spec;

typedef struct {
    unsigned char key[32]; // holds AES-256 full key or DES key
} found_key;

int init_cipher_spec_from_opts(cipher_spec *spec, const void *cli_opts);
int decrypt_buffer(const unsigned char *cipher, int c_len,
                   const unsigned char *key, const unsigned char *iv,
                   unsigned char *plain, const cipher_spec *spec);

// Raw single-block ECB decrypt (no padding), for heuristics
int decrypt_block_raw_ecb(const unsigned char *cipher_block,
                          const unsigned char *key,
                          unsigned char *plain_block,
                          const cipher_spec *spec);

#endif