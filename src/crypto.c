#include "crypto.h"
#include "cli.h"
#include <string.h>
#include <openssl/evp.h>

int init_cipher_spec_from_opts(cipher_spec *spec, const void *cli_opts_void) {
    const cli_options *opts = (const cli_options*)cli_opts_void;
    memset(spec, 0, sizeof(*spec));

    if (strcmp(opts->algo, "AES256")==0) { spec->is_aes = 1; spec->key_len = 32; spec->block_len = 16; }
    else if (strcmp(opts->algo, "DES")==0) { spec->is_aes = 0; spec->key_len = 8; spec->block_len = 8; }
    else return 0;

    if (strcmp(opts->mode, "CBC")==0) spec->use_cbc = 1;
    else if (strcmp(opts->mode,"ECB")==0) spec->use_cbc = 0;
    else return 0;

    spec->iv_from_file = 0;
    if (spec->use_cbc) {
        if (opts->iv_auto) {
            spec->iv_from_file = 1;
        } else if (opts->iv_hex[0]) {
            for (int i=0;i<spec->block_len;i++) {
                unsigned int b; sscanf(opts->iv_hex + 2*i, "%2x", &b);
                spec->iv[i] = (unsigned char)b;
            }
        } else {
            // CBC requires IV; either auto or provided.
            return 0;
        }
    } else {
        memset(spec->iv, 0, sizeof(spec->iv));
    }
    return 1;
}

static const EVP_CIPHER* select_evp(const cipher_spec *spec) {
    if (spec->is_aes) {
        return spec->use_cbc ? EVP_aes_256_cbc() : EVP_aes_256_ecb();
    } else {
        return spec->use_cbc ? EVP_des_cbc() : EVP_des_ecb();
    }
}

int decrypt_buffer(const unsigned char *cipher, int c_len,
                   const unsigned char *key, const unsigned char *iv,
                   unsigned char *plain, const cipher_spec *spec) {
    const EVP_CIPHER *evp = select_evp(spec);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int outlen1=0, outlen2=0;
    int ok = EVP_DecryptInit_ex(ctx, evp, NULL, key, spec->use_cbc ? iv : NULL);
    if (!ok) { EVP_CIPHER_CTX_free(ctx); return -1; }
    EVP_CIPHER_CTX_set_padding(ctx, 1);

    ok = EVP_DecryptUpdate(ctx, plain, &outlen1, cipher, c_len);
    if (!ok) { EVP_CIPHER_CTX_free(ctx); return -2; }
    ok = EVP_DecryptFinal_ex(ctx, plain + outlen1, &outlen2);
    EVP_CIPHER_CTX_free(ctx);
    if (!ok) return -3;
    return outlen1 + outlen2;
}

int decrypt_block_raw_ecb(const unsigned char *cipher_block,
                          const unsigned char *key,
                          unsigned char *plain_block,
                          const cipher_spec *spec) {
    const EVP_CIPHER *evp = spec->is_aes ? EVP_aes_256_ecb() : EVP_des_ecb();

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;
    int ok = EVP_DecryptInit_ex(ctx, evp, NULL, key, NULL);
    if (!ok) { EVP_CIPHER_CTX_free(ctx); return -2; }
    EVP_CIPHER_CTX_set_padding(ctx, 0);

    int outlen = 0;
    ok = EVP_DecryptUpdate(ctx, plain_block, &outlen, cipher_block, spec->block_len);
    if (!ok || outlen != spec->block_len) { EVP_CIPHER_CTX_free(ctx); return -3; }

    int fin = 0;
    ok = EVP_DecryptFinal_ex(ctx, plain_block + outlen, &fin);
    EVP_CIPHER_CTX_free(ctx);
    // No padding expected for a single block in ECB; ignore finalization errors.
    return spec->block_len;
}