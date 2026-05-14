// qwen-codec.cpp: codec CLI for Qwen3-TTS.
//
// Encode a 24 kHz mono WAV into RVQ codes (.rvq), or decode RVQ codes
// back into a 24 kHz mono float32 WAV. Mode is inferred from the input
// file extension: .wav in -> encode, .rvq in -> decode. Output is
// auto-named next to the input file by swapping the extension.
//
// File format (.rvq): flat code stream packed at 11 bits per code,
// LSB-first, no header. Layout is [K, T] row-major. K is fixed by the
// codec config in the GGUF (16 codebooks for the 12Hz tokenizer,
// codebook_size = 2048). T is the frame count derived from filesize.

#include "audio-io.h"
#include "backend.h"
#include "pipeline-codec.h"
#include "utf8.h"
#include "version.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const uint32_t QWEN_RVQ_CODE_MASK = (1u << QWEN_TOKENIZER_CODE_BITS) - 1u;

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", QWEN_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> [-i <input>] [--format <fmt>]\n\n"
            "Required:\n"
            "  --model <gguf>          Codec GGUF (qwen-tokenizer-12hz-*.gguf)\n\n"
            "Optional:\n"
            "  -i <path>               Input. WAV -> encode, .rvq -> decode\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n\n"
            "Output is auto-named next to input : clip.wav -> clip.rvq, clip.rvq -> clip.wav.\n"
            "When -i is omitted, runs a load self-test of the codec GGUF.\n",
            prog);
}

// Symmetric unpack: reads N codes from packed bytes (11 bits LSB-first).
static std::vector<int32_t> unpack_codes(const std::vector<uint8_t> & in, size_t n_codes) {
    std::vector<int32_t> out(n_codes);
    uint64_t             acc         = 0;
    int                  bits_in_acc = 0;
    size_t               in_pos      = 0;
    for (size_t i = 0; i < n_codes; i++) {
        while (bits_in_acc < QWEN_TOKENIZER_CODE_BITS && in_pos < in.size()) {
            acc |= ((uint64_t) in[in_pos++]) << bits_in_acc;
            bits_in_acc += 8;
        }
        out[i] = (int32_t) (acc & QWEN_RVQ_CODE_MASK);
        acc >>= QWEN_TOKENIZER_CODE_BITS;
        bits_in_acc -= QWEN_TOKENIZER_CODE_BITS;
    }
    return out;
}

// Pack flat int32 codes into 11-bit LSB-first packed bytes. Output size is
// ceil(N * 11 / 8) bytes.
static std::vector<uint8_t> pack_codes(const std::vector<int32_t> & codes) {
    const size_t         total_bits = codes.size() * (size_t) QWEN_TOKENIZER_CODE_BITS;
    std::vector<uint8_t> out((total_bits + 7) / 8, 0);
    uint64_t             acc         = 0;
    int                  bits_in_acc = 0;
    size_t               out_pos     = 0;
    for (size_t i = 0; i < codes.size(); i++) {
        acc |= ((uint64_t) ((uint32_t) codes[i] & QWEN_RVQ_CODE_MASK)) << bits_in_acc;
        bits_in_acc += QWEN_TOKENIZER_CODE_BITS;
        while (bits_in_acc >= 8) {
            out[out_pos++] = (uint8_t) (acc & 0xFF);
            acc >>= 8;
            bits_in_acc -= 8;
        }
    }
    if (bits_in_acc > 0) {
        out[out_pos++] = (uint8_t) (acc & 0xFF);
    }
    return out;
}

// Read a .rvq file and unpack it into K*T codes. T is inferred from the
// file size: T = (filesize * 8) / (K * QWEN_TOKENIZER_CODE_BITS).
static bool read_rvq(const char * path, int K, std::vector<int32_t> & codes, int * n_frames) {
    FILE * f = utf8_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[Codec] FATAL: cannot open %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fprintf(stderr, "[Codec] FATAL: %s is empty\n", path);
        fclose(f);
        return false;
    }
    std::vector<uint8_t> buf((size_t) sz);
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        fprintf(stderr, "[Codec] FATAL: short read on %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);

    const size_t total_bits = (size_t) sz * 8;
    const size_t n_codes    = total_bits / (size_t) QWEN_TOKENIZER_CODE_BITS;
    if (n_codes == 0 || (n_codes % (size_t) K) != 0) {
        fprintf(stderr, "[Codec] FATAL: %s yields %zu codes, not a multiple of K=%d\n", path, n_codes, K);
        return false;
    }
    codes     = unpack_codes(buf, n_codes);
    *n_frames = (int) (n_codes / (size_t) K);
    return true;
}

// Pack and write a .rvq file.
static bool write_rvq(const char * path, const std::vector<int32_t> & codes) {
    std::vector<uint8_t> packed = pack_codes(codes);
    FILE *               f      = utf8_fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Codec] FATAL: cannot open %s for write\n", path);
        return false;
    }
    if (fwrite(packed.data(), 1, packed.size(), f) != packed.size()) {
        fprintf(stderr, "[Codec] FATAL: short write on %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

// Replace or append extension on a path string.
static std::string swap_ext(const std::string & path, const char * ext) {
    size_t dot = path.find_last_of('.');
    size_t sep = path.find_last_of("/\\");
    if (dot != std::string::npos && (sep == std::string::npos || dot > sep)) {
        return path.substr(0, dot) + ext;
    }
    return path + ext;
}

// 0: unsupported, 1: encode (.wav in), 2: decode (.rvq in).
static int infer_mode(const char * path) {
    size_t n = strlen(path);
    if (n >= 4 && strcmp(path + n - 4, ".wav") == 0) {
        return 1;
    }
    if (n >= 4 && strcmp(path + n - 4, ".rvq") == 0) {
        return 2;
    }
    return 0;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    if (argc <= 1) {
        print_usage(argv[0]);
        return 0;
    }

    const char * model_path = NULL;
    const char * input_path = NULL;
    WavFormat    wav_fmt    = WAV_S16;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            if (!audio_parse_format(argv[++i], wav_fmt)) {
                fprintf(stderr, "[CLI] ERROR: unknown format: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    int mode = 0;
    if (input_path) {
        mode = infer_mode(input_path);
        if (mode == 0) {
            fprintf(stderr, "[CLI] ERROR: %s: unsupported extension (expect .wav or .rvq)\n", input_path);
            return 1;
        }
    }

    BackendPair bp = backend_init("Codec");
    if (!bp.backend) {
        fprintf(stderr, "[Codec] FATAL: backend init failed\n");
        return 1;
    }

    PipelineCodec pc = {};
    if (!pipeline_codec_load(&pc, model_path, bp)) {
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    int rc = 0;

    if (!input_path) {
        fprintf(stderr, "[Codec] Load self-test passed\n");
    } else if (mode == 1) {
        // Encode .wav -> .rvq
        const std::string out_str = swap_ext(input_path, ".rvq");

        int     T_in     = 0;
        float * audio_in = audio_read_mono(input_path, QWEN_TOKENIZER_SAMPLE_RATE, &T_in);
        if (!audio_in || T_in <= 0) {
            fprintf(stderr, "[Codec] FATAL: cannot read %s\n", input_path);
            free(audio_in);
            rc = 1;
        } else {
            // Pad to a multiple of HOP_LENGTH so the RVQ frame count is integral.
            int hop      = QWEN_TOKENIZER_HOP_LENGTH;
            int T_padded = ((T_in + hop - 1) / hop) * hop;
            int T_frames = T_padded / hop;

            std::vector<float> audio_buf((size_t) T_padded, 0.0f);
            memcpy(audio_buf.data(), audio_in, (size_t) T_in * sizeof(float));
            free(audio_in);

            fprintf(stderr, "[Codec] Encode: %s, %d samples @ %d Hz, padded to %d (%d frames @ 12.5 Hz, %.2f s)\n",
                    input_path, T_in, QWEN_TOKENIZER_SAMPLE_RATE, T_padded, T_frames,
                    (double) T_padded / (double) QWEN_TOKENIZER_SAMPLE_RATE);

            std::vector<int32_t> codes = pipeline_codec_encode(&pc, audio_buf.data(), T_padded);
            if (codes.empty()) {
                fprintf(stderr, "[Codec] FATAL: encode failed\n");
                rc = 1;
            } else if (!write_rvq(out_str.c_str(), codes)) {
                rc = 1;
            } else {
                fprintf(stderr, "[Codec] Wrote %s: K=%d T=%d, %zu codes -> %zu packed bytes\n", out_str.c_str(),
                        QWEN_TOKENIZER_NUM_CODEBOOKS, T_frames, codes.size(),
                        (codes.size() * (size_t) QWEN_TOKENIZER_CODE_BITS + 7) / 8);
            }
        }
    } else {
        // Decode .rvq -> .wav
        const std::string out_str = swap_ext(input_path, ".wav");

        std::vector<int32_t> codes;
        int                  T = 0;
        if (!read_rvq(input_path, QWEN_TOKENIZER_NUM_CODEBOOKS, codes, &T)) {
            rc = 1;
        } else {
            fprintf(stderr, "[Codec] Decode: %s, K=%d T=%d (%.2f s)\n", input_path, QWEN_TOKENIZER_NUM_CODEBOOKS, T,
                    (double) (T * QWEN_TOKENIZER_HOP_LENGTH) / (double) QWEN_TOKENIZER_SAMPLE_RATE);

            std::vector<float> audio = pipeline_codec_decode(&pc, codes.data(), QWEN_TOKENIZER_NUM_CODEBOOKS, T);
            if (audio.empty()) {
                fprintf(stderr, "[Codec] FATAL: decode failed\n");
                rc = 1;
            } else if (!audio_write_wav(out_str.c_str(), audio.data(), (int) audio.size(), QWEN_TOKENIZER_SAMPLE_RATE,
                                        wav_fmt)) {
                fprintf(stderr, "[Codec] FATAL: cannot write %s\n", out_str.c_str());
                rc = 1;
            } else {
                fprintf(stderr, "[Codec] Wrote %s: %d samples @ %d Hz, %.2f s\n", out_str.c_str(), (int) audio.size(),
                        QWEN_TOKENIZER_SAMPLE_RATE, (double) audio.size() / (double) QWEN_TOKENIZER_SAMPLE_RATE);
            }
        }
    }

    pipeline_codec_free(&pc);
    backend_release(bp.backend, bp.cpu_backend);
    return rc;
}
