#include "cli.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void print_usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  ./MyDecryptor --enc <path> --orig <path> --algo AES256|DES --mode ECB|CBC [options]\n\n"
        "Options:\n"
        "  --out <path>             Output decrypted file path\n"
        "  --gpu                    Use CUDA-assisted search\n"
        "  --segment first4|last4|middle4   Variable 32-bit segment within key\n"
        "  --prefix <hex>           AES-256 prefix hex (64 chars)\n"
        "  --suffix <hex>           AES-256 suffix hex (64 chars)\n"
        "  --iv <hex|auto>          IV for CBC (32 hex chars) or 'auto' to use first block of encrypted file\n"
        "  --resume <path>          Resume/progress file\n"
        "  --max <N>                Maximum keys to try (default 4294967296)\n"
        "  --batch <N>              GPU batch size (default 1048576)\n"
    );
}

bool parse_cli(int argc, char **argv, cli_options *out) {
    if (argc < 7) { print_usage(); return false; }
    memset(out, 0, sizeof(*out));
    out->batch = 1048576;
    out->max_keys = 0; // 0 means 2^32
    out->iv_auto = false;

    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--enc") && i+1<argc) out->enc_path = argv[++i];
        else if (!strcmp(argv[i],"--orig") && i+1<argc) out->orig_path = argv[++i];
        else if (!strcmp(argv[i],"--out") && i+1<argc) out->out_path = argv[++i];
        else if (!strcmp(argv[i],"--algo") && i+1<argc) out->algo = argv[++i];
        else if (!strcmp(argv[i],"--mode") && i+1<argc) out->mode = argv[++i];
        else if (!strcmp(argv[i],"--gpu")) out->use_gpu = true;
        else if (!strcmp(argv[i],"--segment") && i+1<argc) out->segment = argv[++i];
        else if (!strcmp(argv[i],"--prefix") && i+1<argc) strncpy(out->prefix_hex, argv[++i], sizeof(out->prefix_hex)-1);
        else if (!strcmp(argv[i],"--suffix") && i+1<argc) strncpy(out->suffix_hex, argv[++i], sizeof(out->suffix_hex)-1);
        else if (!strcmp(argv[i],"--iv") && i+1<argc) {
            const char *val = argv[++i];
            if (!strcmp(val, "auto")) {
                out->iv_auto = true;
                out->iv_hex[0] = 0;
            } else {
                strncpy(out->iv_hex, val, sizeof(out->iv_hex)-1);
                out->iv_auto = false;
            }
        }
        else if (!strcmp(argv[i],"--resume") && i+1<argc) out->resume_file = argv[++i];
        else if (!strcmp(argv[i],"--max") && i+1<argc) out->max_keys = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i],"--batch") && i+1<argc) out->batch = strtoul(argv[++i], NULL, 10);
    }

    if (!out->enc_path || !out->orig_path || !out->algo || !out->mode) { print_usage(); return false; }
    return true;
}