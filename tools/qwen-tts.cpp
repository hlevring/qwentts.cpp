// qwen-tts.cpp : thin CLI wrapper around the Qwen3-TTS synthesis
// pipeline. Parses arguments, loads the talker + codec GGUFs, hands
// off to pipeline_tts_synthesize and writes the resulting waveform as
// a WAV file. All heavy lifting lives in src/pipeline-tts.cpp.
//
// Talker variants : 0.6B-Base / 0.6B-CustomVoice / 1.7B-Base /
// 1.7B-CustomVoice / 1.7B-VoiceDesign. The decoder path is selected
// from GGUF metadata at load time. The CLI surface mirrors the
// omnivoice.cpp tooling : kebab-case flags, --format wav16/wav24/wav32,
// -o '-' streams to stdout, --seed -1 means non deterministic, the
// utterance text comes from --text or stdin if --text is absent.

#include "audio-io.h"
#include "backend.h"
#include "bpe.h"
#include "pipeline-tts.h"
#include "utf8.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", QWEN_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options] -o <out.wav>\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Tokenizer GGUF (qwen-tokenizer-*.gguf)\n"
            "  -o <path>               Output WAV. '-' streams to stdout (pipe friendly).\n\n"
            "Input:\n"
            "  --text <s>              Utterance text. If absent, stdin is read fully.\n\n"
            "Synthesis options:\n"
            "  --lang <name>           Language (auto, english, chinese, ...) (default: english)\n"
            "  --instruct <s>          Style instruction. Required for VoiceDesign, optional\n"
            "                          for CustomVoice. Rejected for Base.\n"
            "  --speaker <name>        Speaker name. Only valid for CustomVoice.\n"
            "  --ref-audio <wav>       Reference WAV path for voice clone (Base only). Mutually\n"
            "                          exclusive with --speaker. Mode A (x_vector_only) extracts\n"
            "                          a speaker embedding via the ECAPA-TDNN encoder.\n"
            "  --ref-text <path>       Path to a UTF-8 text file containing the reference\n"
            "                          transcript for voice clone ICL mode (Base only, requires\n"
            "                          --ref-audio). Switches the prompt to ICL mode B where the\n"
            "                          talker conditions on the reference codec codes.\n"
            "  --max-new <n>           Max new audio frames (default: 2048)\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n\n"
            "Sampling options:\n"
            "  --seed <n>              Sampling seed, -1 for random (default: -1)\n"
            "  --greedy                Disable stochastic sampling on both stacks\n"
            "  --temp <f>              Talker temperature (default: 0.9)\n"
            "  --top-k <n>             Talker top-k (default: 50, 0 = disabled)\n"
            "  --top-p <f>             Talker top-p (default: 1.0)\n"
            "  --rep-pen <f>           Talker repetition penalty (default: 1.05)\n"
            "  --sub-temp <f>          Sub-talker temperature (default: 0.9)\n"
            "  --sub-top-k <n>         Sub-talker top-k (default: 50)\n"
            "  --sub-top-p <f>         Sub-talker top-p (default: 1.0)\n\n"
            "Debug:\n"
            "  --dump <dir>            Dump intermediate tensors for cossim debug\n",
            prog);
}

struct Args {
    const char * model;
    const char * codec;
    const char * text;
    const char * lang;
    const char * instruct;
    const char * speaker;
    const char * ref_audio;
    const char * ref_text_path;
    const char * dump_dir;
    const char * out_wav;
    const char * format;
    int          max_new_tokens;
    int64_t      seed;
    bool         do_sample;
    float        temperature;
    int          top_k;
    float        top_p;
    float        repetition_penalty;
    int          subtalker_top_k;
    float        subtalker_top_p;
    float        subtalker_temperature;
    bool         subtalker_do_sample;
};

// Read all of stdin into a string. Trims trailing newlines so a piped
// text file behaves like a clean --text argument.
static std::string read_stdin_text() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

// Read a small text file into a string. Trims trailing newlines.
static bool read_text_file(const char * path, std::string & out) {
    FILE * f = utf8_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[CLI] FATAL: cannot open '%s'\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        fprintf(stderr, "[CLI] FATAL: ftell failed on '%s'\n", path);
        return false;
    }
    out.resize((size_t) sz);
    if (sz > 0 && fread(&out[0], 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f);
        fprintf(stderr, "[CLI] FATAL: fread failed on '%s'\n", path);
        return false;
    }
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

static bool parse_args(int argc, char ** argv, Args & a) {
    a                       = {};
    a.lang                  = "english";
    a.format                = "wav16";
    a.max_new_tokens        = 2048;
    a.seed                  = -1;
    a.do_sample             = true;
    a.temperature           = 0.9f;
    a.top_k                 = 50;
    a.top_p                 = 1.0f;
    a.repetition_penalty    = 1.05f;
    a.subtalker_do_sample   = true;
    a.subtalker_top_k       = 50;
    a.subtalker_top_p       = 1.0f;
    a.subtalker_temperature = 0.9f;
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            return false;
        }
        if (std::strcmp(arg, "--model") == 0 && i + 1 < argc) {
            a.model = argv[++i];
        } else if (std::strcmp(arg, "--codec") == 0 && i + 1 < argc) {
            a.codec = argv[++i];
        } else if (std::strcmp(arg, "--text") == 0 && i + 1 < argc) {
            a.text = argv[++i];
        } else if (std::strcmp(arg, "--lang") == 0 && i + 1 < argc) {
            a.lang = argv[++i];
        } else if (std::strcmp(arg, "--instruct") == 0 && i + 1 < argc) {
            a.instruct = argv[++i];
        } else if (std::strcmp(arg, "--speaker") == 0 && i + 1 < argc) {
            a.speaker = argv[++i];
        } else if (std::strcmp(arg, "--ref-audio") == 0 && i + 1 < argc) {
            a.ref_audio = argv[++i];
        } else if (std::strcmp(arg, "--ref-text") == 0 && i + 1 < argc) {
            a.ref_text_path = argv[++i];
        } else if (std::strcmp(arg, "--format") == 0 && i + 1 < argc) {
            a.format = argv[++i];
        } else if (std::strcmp(arg, "--dump") == 0 && i + 1 < argc) {
            a.dump_dir = argv[++i];
        } else if (std::strcmp(arg, "--max-new") == 0 && i + 1 < argc) {
            a.max_new_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            a.seed = (int64_t) std::atoll(argv[++i]);
        } else if (std::strcmp(arg, "--greedy") == 0) {
            // Greedy mode : argmax sampling on both stacks. The sampling
            // fast path in sampling.h uses temperature <= 0 to short
            // circuit to argmax, bypassing rep penalty and top-k/p
            // truncation, which exactly mirrors the Python reference
            // greedy behaviour used by tests/debug-tts-cossim.py.
            a.do_sample           = false;
            a.subtalker_do_sample = false;
        } else if (std::strcmp(arg, "--temp") == 0 && i + 1 < argc) {
            a.temperature = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            a.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            a.top_p = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--rep-pen") == 0 && i + 1 < argc) {
            a.repetition_penalty = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--sub-temp") == 0 && i + 1 < argc) {
            a.subtalker_temperature = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--sub-top-k") == 0 && i + 1 < argc) {
            a.subtalker_top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--sub-top-p") == 0 && i + 1 < argc) {
            a.subtalker_top_p = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "-o") == 0 && i + 1 < argc) {
            a.out_wav = argv[++i];
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown or incomplete argument: %s\n", arg);
            return false;
        }
    }
    return a.model && a.codec;
}

static int run(const Args & a) {
    BackendPair bp = backend_init("Talker");

    PipelineTTS pt = {};
    if (!pipeline_tts_load(&pt, a.model, a.codec, bp)) {
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    // Validate mode flag combination against the loaded model_type. The
    // upstream Python raises ValueError when generate_voice_design is
    // called on a non voice_design model and the same shape applies to
    // generate_custom_voice. We mirror that here, explicit and KISS, so
    // the user never gets a silently wrong synthesis.
    const std::string mt = pt.model_type;
    if (a.speaker && mt != "custom_voice") {
        fprintf(stderr, "[CLI] ERROR: --speaker is only valid for custom_voice models (loaded: %s)\n", mt.c_str());
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }
    if (a.instruct && mt == "base") {
        fprintf(stderr, "[CLI] ERROR: --instruct is not supported for base models\n");
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }
    if (mt == "custom_voice" && !a.speaker) {
        fprintf(stderr, "[CLI] ERROR: custom_voice models require --speaker\n");
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }
    if (mt == "voice_design" && (!a.instruct || a.instruct[0] == '\0')) {
        fprintf(stderr, "[CLI] ERROR: voice_design models require --instruct\n");
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }
    if (a.ref_audio && mt != "base") {
        fprintf(stderr, "[CLI] ERROR: --ref-audio is only valid for base models (loaded: %s)\n", mt.c_str());
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }
    if (a.speaker && a.ref_audio) {
        fprintf(stderr, "[CLI] ERROR: --speaker and --ref-audio are mutually exclusive\n");
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }
    if (a.ref_text_path && !a.ref_audio) {
        fprintf(stderr, "[CLI] ERROR: --ref-text requires --ref-audio\n");
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    // Load the reference transcript from the file passed via --ref-text.
    // Empty content is rejected since the ICL prompt needs it non empty.
    std::string  ref_text_buf;
    const char * ref_text = NULL;
    if (a.ref_text_path) {
        if (!read_text_file(a.ref_text_path, ref_text_buf)) {
            pipeline_tts_free(&pt);
            backend_release(bp.backend, bp.cpu_backend);
            return 1;
        }
        if (ref_text_buf.empty()) {
            fprintf(stderr, "[CLI] ERROR: --ref-text file '%s' is empty\n", a.ref_text_path);
            pipeline_tts_free(&pt);
            backend_release(bp.backend, bp.cpu_backend);
            return 1;
        }
        ref_text = ref_text_buf.c_str();
    }

    // Decode the reference WAV once, mono at the codec sample rate. The
    // pipeline consumes the buffer directly so the WAV is read exactly
    // once regardless of how many encoders need the audio (speaker
    // encoder embedding + codec encoder RVQ codes for ICL mode B).
    std::vector<float>                       ref_audio_buf;
    std::unique_ptr<float, void (*)(void *)> raw_holder(NULL, std::free);
    const float *                            ref_audio_24k = NULL;
    int                                      ref_n_samples = 0;
    if (a.ref_audio) {
        int     T_in = 0;
        float * raw  = audio_read_mono(a.ref_audio, QWEN_TOKENIZER_SAMPLE_RATE, &T_in);
        if (!raw || T_in <= 0) {
            fprintf(stderr, "[CLI] ERROR: cannot read --ref-audio '%s'\n", a.ref_audio);
            if (raw) {
                std::free(raw);
            }
            pipeline_tts_free(&pt);
            backend_release(bp.backend, bp.cpu_backend);
            return 1;
        }
        raw_holder.reset(raw);
        ref_audio_24k = raw;
        ref_n_samples = T_in;
    }

    // Resolve output WAV format string : wav16 / wav24 / wav32. Default
    // wav16 mirrors the omnivoice.cpp default.
    WavFormat wav_fmt;
    if (!audio_parse_format(a.format, wav_fmt)) {
        fprintf(stderr, "[CLI] ERROR: invalid --format '%s' (expected wav16, wav24, wav32)\n", a.format);
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    // Resolve utterance text : explicit --text wins, otherwise read stdin
    // fully. Empty stdin combined with no --text triggers a clean error.
    std::string  text_buf;
    const char * text = a.text;
    if (!text) {
        text_buf = read_stdin_text();
        if (text_buf.empty()) {
            fprintf(stderr, "[CLI] ERROR: no --text and stdin is empty\n");
            pipeline_tts_free(&pt);
            backend_release(bp.backend, bp.cpu_backend);
            return 1;
        }
        text = text_buf.c_str();
    }

    // Resolve seed : -1 means non deterministic, sample from a hardware
    // random_device. Anything else is taken verbatim, including negative
    // values reaching int64 range, so reproducibility is one --seed away.
    int64_t seed = a.seed;
    if (seed < 0) {
        std::random_device rd;
        seed = (int64_t) (((uint64_t) rd() << 32) ^ (uint64_t) rd());
    }

    BPETokenizer tok = {};
    if (!load_bpe_from_gguf(&tok, a.model)) {
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }
    const char * specials_keys[] = {
        "qwen3-tts.text.im_start_id", "qwen3-tts.text.im_end_id",  "qwen3-tts.text.tts_pad_id",
        "qwen3-tts.text.tts_bos_id",  "qwen3-tts.text.tts_eos_id",
    };
    bpe_load_specials_from_keys(&tok, a.model, specials_keys, 5);

    PipelineTTSSynthesizeParams p = {};
    p.text                        = text;
    p.lang                        = a.lang;
    p.instruct                    = a.instruct;
    p.speaker                     = a.speaker;
    p.ref_audio_24k               = ref_audio_24k;
    p.ref_n_samples               = ref_n_samples;
    p.ref_text                    = ref_text;
    p.seed                        = seed;
    p.max_new_tokens              = a.max_new_tokens;
    p.do_sample                   = a.do_sample;
    p.temperature                 = a.temperature;
    p.top_k                       = a.top_k;
    p.top_p                       = a.top_p;
    p.repetition_penalty          = a.repetition_penalty;
    p.subtalker_do_sample         = a.subtalker_do_sample;
    p.subtalker_temperature       = a.subtalker_temperature;
    p.subtalker_top_k             = a.subtalker_top_k;
    p.subtalker_top_p             = a.subtalker_top_p;
    p.dump_dir                    = a.dump_dir;

    PipelineTTSSynthesizeOutput out;
    if (!pipeline_tts_synthesize(&pt, &tok, p, &out)) {
        pipeline_tts_free(&pt);
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    if (!out.audio.empty()) {
        const char * out_path = a.out_wav ? a.out_wav : "out.wav";
        if (!audio_write_wav(out_path, out.audio.data(), (int) out.audio.size(), out.sample_rate, wav_fmt)) {
            fprintf(stderr, "[Pipeline] FATAL: WAV write failed for %s\n", out_path);
            pipeline_tts_free(&pt);
            backend_release(bp.backend, bp.cpu_backend);
            return 1;
        }
        qt_log(QT_LOG_INFO, "[Pipeline] Wrote %zu samples (%.2f s) -> %s", out.audio.size(),
               (double) out.audio.size() / (double) out.sample_rate, out_path);
    }

    pipeline_tts_free(&pt);
    backend_release(bp.backend, bp.cpu_backend);
    return 0;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    Args a;
    if (!parse_args(argc, argv, a)) {
        print_usage(argv[0]);
        return 1;
    }
    try {
        return run(a);
    } catch (const std::runtime_error & e) {
        qt_set_error("%s", e.what());
        qt_log(QT_LOG_ERROR, "%s", e.what());
        return 1;
    }
}
