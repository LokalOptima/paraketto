// weights.cpp — Weight loading shared by FP16 and FP8 backends

// weights.cpp — Weight loading shared by FP16 and FP8 backends
//
// Loads weights.bin into a single contiguous GPU allocation and assigns
// struct field pointers. Linked by paraketto.cuda, .cublas, and .fp8.

#include "conformer.h"
#include "common.h"
#include "kernels.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helper: align up
// ---------------------------------------------------------------------------

static size_t align_up(size_t x, size_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Parse header
// ---------------------------------------------------------------------------

static std::vector<TensorDesc> parse_header(const char* header_text, size_t header_len) {
    std::vector<TensorDesc> tensors;
    std::string text(header_text, header_len);
    std::istringstream iss(text);
    std::string line;

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        TensorDesc td;
        ls >> td.name >> td.offset >> td.size_bytes >> td.dtype;
        int d;
        while (ls >> d) td.shape.push_back(d);
        tensors.push_back(std::move(td));
    }
    return tensors;
}

// ---------------------------------------------------------------------------
// Weights: pointer assignment (shared by load and upload)
// ---------------------------------------------------------------------------

static void assign_weight_pointers(Weights& w) {
    uint8_t* gpu_base = (uint8_t*)w.gpu_data;

    auto lookup = [&](const std::string& name) -> half* {
        auto it = w.name_to_idx.find(name);
        if (it == w.name_to_idx.end()) return nullptr;
        return (half*)(gpu_base + w.tensors[it->second].offset);
    };

    // Subsampling (pre_encode)
    for (int i : {0, 2, 3, 5, 6}) {
        std::string pre = "encoder/pre_encode.conv." + std::to_string(i);
        w.sub_conv[i].weight = lookup(pre + ".weight");
        w.sub_conv[i].bias   = lookup(pre + ".bias");
    }
    w.sub_out_w = lookup("encoder/pre_encode.out.weight");
    w.sub_out_b = lookup("encoder/pre_encode.out.bias");

    // Conformer blocks (x24)
    for (int i = 0; i < 24; i++) {
        auto& blk = w.blocks[i];
        std::string pre = "encoder/layers." + std::to_string(i);

        blk.ff1_ln_w = lookup(pre + ".norm_feed_forward1.weight");
        blk.ff1_ln_b = lookup(pre + ".norm_feed_forward1.bias");
        blk.ff1_w1   = lookup(pre + ".feed_forward1.linear1.weight");
        blk.ff1_w2   = lookup(pre + ".feed_forward1.linear2.weight");

        blk.mhsa_ln_w = lookup(pre + ".norm_self_att.weight");
        blk.mhsa_ln_b = lookup(pre + ".norm_self_att.bias");
        blk.q_w       = lookup(pre + ".self_attn.linear_q.weight");
        blk.k_w       = lookup(pre + ".self_attn.linear_k.weight");
        blk.v_w       = lookup(pre + ".self_attn.linear_v.weight");
        blk.pos_w     = lookup(pre + ".self_attn.linear_pos.weight");
        blk.pos_bias_u = lookup(pre + ".self_attn.pos_bias_u");
        blk.pos_bias_v = lookup(pre + ".self_attn.pos_bias_v");
        blk.out_w     = lookup(pre + ".self_attn.linear_out.weight");

        blk.conv_ln_w  = lookup(pre + ".norm_conv.weight");
        blk.conv_ln_b  = lookup(pre + ".norm_conv.bias");
        blk.conv_pw1_w = lookup(pre + ".conv.pointwise_conv1.weight");
        blk.conv_dw_w  = lookup(pre + ".conv.depthwise_conv.weight");
        blk.conv_dw_b  = lookup(pre + ".conv.depthwise_conv.bias");
        blk.conv_pw2_w = lookup(pre + ".conv.pointwise_conv2.weight");

        blk.ff2_ln_w = lookup(pre + ".norm_feed_forward2.weight");
        blk.ff2_ln_b = lookup(pre + ".norm_feed_forward2.bias");
        blk.ff2_w1   = lookup(pre + ".feed_forward2.linear1.weight");
        blk.ff2_w2   = lookup(pre + ".feed_forward2.linear2.weight");

        blk.final_ln_w = lookup(pre + ".norm_out.weight");
        blk.final_ln_b = lookup(pre + ".norm_out.bias");
    }

    // Decoder: embedding + LSTM
    w.embed_w = lookup("decoder/decoder.prediction.embed.weight");
    w.lstm0_w_ih = lookup("decoder/decoder.dec_rnn.lstm.weight_ih");
    w.lstm0_w_hh = lookup("decoder/decoder.dec_rnn.lstm.weight_hh");
    w.lstm0_bias = lookup("decoder/decoder.dec_rnn.lstm.bias");
    w.lstm1_w_ih = lookup("decoder/decoder.dec_rnn.lstm.1.weight_ih");
    w.lstm1_w_hh = lookup("decoder/decoder.dec_rnn.lstm.1.weight_hh");
    w.lstm1_bias = lookup("decoder/decoder.dec_rnn.lstm.1.bias");

    // Joint network
    w.enc_proj_w = lookup("decoder/joint.enc.weight");
    w.enc_proj_b = lookup("decoder/joint.enc.bias");
    w.dec_proj_w = lookup("decoder/joint.pred.weight");
    w.dec_proj_b = lookup("decoder/joint.pred.bias");
    w.out_proj_w = lookup("decoder/joint.joint_net.joint_net.2.weight");
    w.out_proj_b = lookup("decoder/joint.joint_net.2.bias");
}

// ---------------------------------------------------------------------------
// Weights::prefetch — CPU only, no CUDA. mmap + populate pages + parse header.
// ---------------------------------------------------------------------------

Weights Weights::prefetch(const std::string& path, bool populate) {
    Weights w;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open weights: %s\n", path.c_str());
        std::exit(1);
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    int flags = MAP_PRIVATE | (populate ? MAP_POPULATE : 0);
    void* mapped = mmap(nullptr, file_size, PROT_READ, flags, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", path.c_str());
        std::exit(1);
    }
    if (populate) madvise(mapped, file_size, MADV_SEQUENTIAL);
    const uint8_t* base = (const uint8_t*)mapped;

    // Parse file header
    uint32_t magic;
    memcpy(&magic, base, 4);
    if (magic != PRKT_MAGIC) {
        fprintf(stderr, "Bad magic in %s: expected PRKT\n", path.c_str());
        munmap(mapped, file_size);
        std::exit(1);
    }

    uint32_t version;
    memcpy(&version, base + 4, 4);
    if (version != PRKT_VERSION) {
        fprintf(stderr, "Unsupported weight file version %u (expected %u)\n", version, PRKT_VERSION);
        munmap(mapped, file_size);
        std::exit(1);
    }

    uint64_t header_len;
    memcpy(&header_len, base + 8, 8);
    w.tensors = parse_header((const char*)(base + 16), header_len);

    for (size_t i = 0; i < w.tensors.size(); i++)
        w.name_to_idx[w.tensors[i].name] = i;

    size_t header_end = 16 + header_len;
    size_t data_start = align_up(header_end, HEADER_ALIGN);

    size_t total_data = 0;
    for (auto& td : w.tensors) {
        size_t end = td.offset + td.size_bytes;
        if (end > total_data) total_data = end;
    }
    if (!w.tensors.empty()) {
        auto& last = w.tensors.back();
        total_data = std::max(total_data, align_up(last.offset + last.size_bytes, 256));
    }

    w.gpu_data_size = total_data;
    w.mmap_ptr = mapped;
    w.mmap_size = file_size;
    w.data_offset = data_start;

    return w;
}

// ---------------------------------------------------------------------------
// Weights::from_embedded — parse header from in-memory data (no mmap).
// ---------------------------------------------------------------------------

Weights Weights::from_embedded(const uint8_t* data, size_t size) {
    Weights w;
    const uint8_t* base = data;

    uint32_t magic;
    memcpy(&magic, base, 4);
    if (magic != PRKT_MAGIC) {
        fprintf(stderr, "Bad magic in embedded weights\n");
        std::exit(1);
    }

    uint32_t version;
    memcpy(&version, base + 4, 4);
    if (version != PRKT_VERSION) {
        fprintf(stderr, "Unsupported embedded weight version %u (expected %u)\n", version, PRKT_VERSION);
        std::exit(1);
    }

    uint64_t header_len;
    memcpy(&header_len, base + 8, 8);
    w.tensors = parse_header((const char*)(base + 16), header_len);

    for (size_t i = 0; i < w.tensors.size(); i++)
        w.name_to_idx[w.tensors[i].name] = i;

    size_t header_end = 16 + header_len;
    size_t data_start = align_up(header_end, HEADER_ALIGN);

    size_t total_data = 0;
    for (auto& td : w.tensors) {
        size_t end = td.offset + td.size_bytes;
        if (end > total_data) total_data = end;
    }
    if (!w.tensors.empty()) {
        auto& last = w.tensors.back();
        total_data = std::max(total_data, align_up(last.offset + last.size_bytes, 256));
    }

    w.gpu_data_size = total_data;
    w.mmap_ptr = nullptr;       // not mmap'd — don't munmap
    w.mmap_size = 0;
    w.data_offset = data_start;
    w.embedded_ptr = data;       // store for upload

    return w;
}

// ---------------------------------------------------------------------------
// Weights::allocate_only — cudaMalloc without data copy (FP8 path: data comes
// from weights_fp8.bin which is self-contained).
// ---------------------------------------------------------------------------

void Weights::allocate_only() {
    CUDA_CHECK(cudaMalloc(&gpu_data, gpu_data_size));
    if (mmap_ptr) {
        munmap(mmap_ptr, mmap_size);
        mmap_ptr = nullptr;
        mmap_size = 0;
    }
    embedded_ptr = nullptr;
    data_offset = 0;
    assign_weight_pointers(*this);
}

// ---------------------------------------------------------------------------
// Weights::upload — cudaMalloc + cudaMemcpy from prefetched mmap, assign ptrs.
// ---------------------------------------------------------------------------

void Weights::upload(cudaStream_t stream) {
    const uint8_t* base = embedded_ptr ? embedded_ptr : (const uint8_t*)mmap_ptr;

    CUDA_CHECK(cudaMalloc(&gpu_data, gpu_data_size));

    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_data, base + data_offset, gpu_data_size,
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(gpu_data, base + data_offset, gpu_data_size,
                               cudaMemcpyHostToDevice));
    }

    if (mmap_ptr) {
        munmap(mmap_ptr, mmap_size);
        mmap_ptr = nullptr;
        mmap_size = 0;
    }
    embedded_ptr = nullptr;
    data_offset = 0;

    assign_weight_pointers(*this);
}

// ---------------------------------------------------------------------------
// Weights::load — convenience: prefetch + upload in one call.
// ---------------------------------------------------------------------------

Weights Weights::load(const std::string& path, cudaStream_t stream) {
    Weights w = prefetch(path);
    w.upload(stream);
    return w;
}

// ---------------------------------------------------------------------------
// Weights::free
// ---------------------------------------------------------------------------

void Weights::free() {
    if (gpu_data) {
        cudaFree(gpu_data);
        gpu_data = nullptr;
        gpu_data_size = 0;
    }
}

// ---------------------------------------------------------------------------
// Weights::get
// ---------------------------------------------------------------------------

half* Weights::get(const std::string& name) const {
    auto it = name_to_idx.find(name);
    if (it == name_to_idx.end()) return nullptr;
    return (half*)((uint8_t*)gpu_data + tensors[it->second].offset);
}

// ---------------------------------------------------------------------------
// Weights::print_info
// ---------------------------------------------------------------------------

void Weights::print_info() const {
    fprintf(stderr, "weights: %zu tensors, %.1f MB GPU\n",
            tensors.size(), gpu_data_size / (1024.0 * 1024.0));

    // Check key struct fields are populated
    int missing = 0;
    auto check = [&](const char* label, const half* ptr) {
        if (!ptr) {
            fprintf(stderr, "  WARNING: %s not found in weight file\n", label);
            missing++;
        }
    };

    // Subsampling
    check("sub_conv[0].weight", sub_conv[0].weight);
    check("sub_conv[6].weight", sub_conv[6].weight);
    check("sub_out_w", sub_out_w);

    // Conformer blocks (spot check first and last)
    check("blocks[0].ff1_w1", blocks[0].ff1_w1);
    check("blocks[0].q_w", blocks[0].q_w);
    check("blocks[0].k_w", blocks[0].k_w);
    check("blocks[0].v_w", blocks[0].v_w);
    check("blocks[0].conv_dw_w", blocks[0].conv_dw_w);
    check("blocks[23].ff2_w2", blocks[23].ff2_w2);
    check("blocks[23].final_ln_w", blocks[23].final_ln_w);

    // Decoder
    check("embed_w", embed_w);
    check("lstm0_w_ih", lstm0_w_ih);
    check("lstm0_bias", lstm0_bias);
    check("lstm1_w_ih", lstm1_w_ih);
    check("lstm1_bias", lstm1_bias);

    // Joint
    check("enc_proj_w", enc_proj_w);
    check("dec_proj_w", dec_proj_w);
    check("out_proj_w", out_proj_w);
    check("out_proj_b", out_proj_b);

    if (missing > 0) {
        fprintf(stderr, "  %d key weights missing — run 'make inspect-onnx' to check names\n", missing);
    } else {
        fprintf(stderr, "  all key weights mapped successfully\n");
    }
}

// =========================================================================
// CudaModel — encoder + decoder forward pass
// =========================================================================

// ---------------------------------------------------------------------------
// CudaModel::init
// ---------------------------------------------------------------------------

