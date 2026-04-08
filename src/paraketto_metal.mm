// paraketto_metal.mm — Metal backend entry point for Parakeet TDT 0.6B
//
// Build: make paraketto.metal
// Usage: ./paraketto.metal [--weights FILE] audio.wav

#import <Metal/Metal.h>

#include "conformer_metal.h"
#include "metal_context.h"
#include "common_metal.h"
#include "mel_data.h"   // filterbank + Hann window (no CUDA)
#include "vocab.h"      // tokenizer (no CUDA)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// WAV loader (inline, no CUDA dependency)
// ---------------------------------------------------------------------------

struct WavData { std::vector<float> samples; int sample_rate; };

static WavData read_wav(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); exit(1); }
    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4)) { fprintf(stderr, "Not WAV: %s\n", path.c_str()); exit(1); }
    fseek(f, 4, SEEK_CUR);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4)) { fprintf(stderr, "Not WAVE\n"); exit(1); }

    int fmt_type=0, channels=0, sample_rate=0, bits=0;
    uint32_t data_size=0;
    while (!feof(f)) {
        char id[4]; uint32_t sz;
        if (fread(id,1,4,f)!=4) break;
        if (fread(&sz,4,1,f)!=1) break;
        if (!memcmp(id,"fmt ",4)) {
            uint16_t fmt,ch,ba,bps; uint32_t sr,br;
            fread(&fmt,2,1,f); fread(&ch,2,1,f); fread(&sr,4,1,f);
            fread(&br,4,1,f); fread(&ba,2,1,f); fread(&bps,2,1,f);
            fmt_type=fmt; channels=ch; sample_rate=sr; bits=bps;
            if(sz>16) fseek(f,sz-16,SEEK_CUR);
        } else if (!memcmp(id,"data",4)) { data_size=sz; break; }
        else fseek(f,sz,SEEK_CUR);
    }
    if (channels!=1) { fprintf(stderr,"Need mono WAV\n"); exit(1); }

    WavData wav; wav.sample_rate = sample_rate;
    if (fmt_type==1 && bits==16) {
        int n = data_size/2; wav.samples.resize(n);
        std::vector<int16_t> raw(n); fread(raw.data(),2,n,f);
        for(int i=0;i<n;i++) wav.samples[i]=raw[i]/32768.0f;
    } else if (fmt_type==3 && bits==32) {
        int n = data_size/4; wav.samples.resize(n);
        fread(wav.samples.data(),4,n,f);
    } else { fprintf(stderr,"Unsupported WAV format\n"); exit(1); }
    fclose(f);

    if (sample_rate==24000) {
        int out_n=(int)((int64_t)wav.samples.size()*16000/24000);
        std::vector<float> out(out_n);
        for(int i=0;i<out_n;i++){
            double s=(double)i*24000.0/16000.0; int s0=(int)s; double fr=s-s0;
            float v0=(s0<(int)wav.samples.size())?wav.samples[s0]:0;
            float v1=(s0+1<(int)wav.samples.size())?wav.samples[s0+1]:v0;
            out[i]=v0+(float)fr*(v1-v0);
        }
        wav.samples=std::move(out); wav.sample_rate=16000;
    }
    return wav;
}

// ---------------------------------------------------------------------------
// CPU Mel spectrogram (uses mel_data.h tables)
// ---------------------------------------------------------------------------

struct MetalMelSpec {
    std::vector<float> preemph_buf, frames_buf;

    void compute(const float* audio, int num_samples,
                 char* pool, size_t mel_fp32_off,
                 int& n_frames, int& n_valid) {
        if (num_samples <= 0) { n_frames=0; n_valid=0; return; }

        if ((int)preemph_buf.size() < num_samples) preemph_buf.resize(num_samples);
        preemph_buf[0] = audio[0];
        for (int i=1; i<num_samples; i++)
            preemph_buf[i] = audio[i] - PREEMPH * audio[i-1];

        int pad = N_FFT/2;
        n_frames = (num_samples + 2*pad - N_FFT) / HOP + 1;
        n_valid  = num_samples / HOP;

        size_t need = (size_t)n_frames * N_FFT;
        if (frames_buf.size() < need) frames_buf.resize(need);

        for (int f=0; f<n_frames; f++) {
            float* row = frames_buf.data() + (size_t)f*N_FFT;
            int base = f*HOP - pad;
            for (int i=0; i<N_FFT; i++) {
                int pos = base+i;
                row[i] = (pos>=0 && pos<num_samples ? preemph_buf[pos] : 0.0f) * HANN_WINDOW[i];
            }
        }

        // CPU FFT + mel filterbank + log
        std::vector<float> mel_log((size_t)n_frames * N_MELS);
        for (int fr=0; fr<n_frames; fr++) {
            const float* in = frames_buf.data() + (size_t)fr*N_FFT;
            float sr[512], si[512];
            for (int i=0; i<512; i++) {
                int j=i, rev=0;
                for (int b=0; b<9; b++) { rev=(rev<<1)|(j&1); j>>=1; }
                sr[rev]=in[i]; si[rev]=0;
            }
            for (int s=0; s<9; s++) {
                int sz=1<<(s+1), hs=sz>>1;
                for (int g=0; g<512/sz; g++)
                    for (int k=0; k<hs; k++) {
                        int a=g*sz+k, b=a+hs;
                        float ang=-6.283185307179586f*k/sz;
                        float wr=cosf(ang), wi=sinf(ang);
                        float tr=wr*sr[b]-wi*si[b], ti=wr*si[b]+wi*sr[b];
                        float ar=sr[a], ai=si[a];
                        sr[a]=ar+tr; si[a]=ai+ti; sr[b]=ar-tr; si[b]=ai-ti;
                    }
            }
            float mel_acc[128]={};
            for (int i=0; i<N_MEL_ENTRIES; i++)
                mel_acc[MEL_FILTERBANK[i].mel] +=
                    (sr[MEL_FILTERBANK[i].freq]*sr[MEL_FILTERBANK[i].freq]+
                     si[MEL_FILTERBANK[i].freq]*si[MEL_FILTERBANK[i].freq]) *
                    MEL_FILTERBANK[i].weight;
            for (int i=0; i<N_MELS; i++)
                mel_log[fr*N_MELS+i] = logf(mel_acc[i]+LOG_EPS);
        }

        // Normalize + transpose → [128, n_valid]
        float* mel_ptr = (float*)(pool + mel_fp32_off);
        for (int ch=0; ch<N_MELS; ch++) {
            float sum=0;
            for (int i=0; i<n_valid; i++) sum += mel_log[i*N_MELS+ch];
            float mean = sum/n_valid;
            float vs=0;
            for (int i=0; i<n_valid; i++) { float d=mel_log[i*N_MELS+ch]-mean; vs+=d*d; }
            float inv = 1.0f/(sqrtf(vs/std::max(1,n_valid-1))+1e-5f);
            for (int i=0; i<n_valid; i++)
                mel_ptr[ch*n_valid+i] = (mel_log[i*N_MELS+ch]-mean)*inv;
        }
    }
};

// ---------------------------------------------------------------------------
// Auto-download
// ---------------------------------------------------------------------------

static std::string cache_dir() {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg&&xdg[0]) return std::string(xdg)+"/paraketto";
    const char* home = getenv("HOME");
    if (home&&home[0]) return std::string(home)+"/.cache/paraketto";
    return ".";
}

static void mkdirs(const std::string& p) {
    std::string c;
    for (auto ch : p) { c+=ch; if(ch=='/') mkdir(c.c_str(),0755); }
    if (!c.empty()&&c.back()!='/') mkdir(c.c_str(),0755);
}

static void ensure_file(const std::string& path, const char* url) {
    if (access(path.c_str(),F_OK)==0) return;
    auto d=path.substr(0,path.rfind('/')); if(!d.empty()) mkdirs(d);
    fprintf(stderr,"Downloading %s\n",path.c_str());
    pid_t pid=fork();
    if(pid==0){execlp("curl","curl","-#","-fL","-o",path.c_str(),url,nullptr);_exit(127);}
    int st; waitpid(pid,&st,0);
    if(!WIFEXITED(st)||WEXITSTATUS(st)!=0){unlink(path.c_str());fprintf(stderr,"Download failed\n");exit(1);}
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

struct Pipeline {
    MetalModel model;
    MetalMelSpec mel;
    double last_mel_ms=0, last_enc_ms=0, last_dec_ms=0;
    bool profile=false;

    void init(const std::string& wp) { model.init(wp.c_str(), MAX_MEL_FRAMES); }

    std::string transcribe(const float* s, int n) {
        if(n<HOP*10) return "";
        auto t0=std::chrono::high_resolution_clock::now();
        int nf, nv;
        char* pool=(char*)model.ctx.buffer_contents(model.gpu_pool_handle);
        mel.compute(s, n, pool, model.mel_fp32_off, nf, nv);
        auto t1=std::chrono::high_resolution_clock::now();
        int T = profile ? model.encode_gpu_profile(nv) : model.encode_gpu(nv);
        auto t2=std::chrono::high_resolution_clock::now();
        auto text=decode(T);
        auto t3=std::chrono::high_resolution_clock::now();
        last_mel_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
        last_enc_ms=std::chrono::duration<double,std::milli>(t2-t1).count();
        last_dec_ms=std::chrono::duration<double,std::milli>(t3-t2).count();
        return text;
    }

private:
    std::string decode(int enc_len) {
        model.decoder_reset();
        std::vector<int> tokens;
        int last_token=model.blank_id;
        int t=0, emitted=0;
        char* pool=(char*)model.ctx.buffer_contents(model.gpu_pool_handle);
        while(t<enc_len){
            model.decode_step(t,last_token);
            int* a=(int*)(pool+model.argmax_out_off);
            int token=a[0], step=a[1];
            if(token!=model.blank_id){
                model.decoder_commit(); tokens.push_back(token);
                last_token=token; emitted++;
            }
            if(step>0){t+=step;emitted=0;}
            else if(token==model.blank_id||emitted>=2){t++;emitted=0;}
        }
        return detokenize(tokens,VOCAB_V2,model.n_vocab);
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if(argc<2){fprintf(stderr,"Usage: %s [--weights FILE] audio.wav\n",argv[0]);return 1;}

    std::string dir=cache_dir();
    std::string wp=dir+"/paraketto-fp16.bin";
    std::vector<std::string> wavs;
    bool profile=false;

    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--weights"&&i+1<argc) wp=argv[++i];
        else if(a=="--profile") profile=true;
        else if(a=="-h"||a=="--help"){fprintf(stderr,"Usage: %s [--weights F] [--profile] wav...\n",argv[0]);return 0;}
        else wavs.push_back(a);
    }
    if(wavs.empty()){fprintf(stderr,"No WAV files.\n");return 1;}

    ensure_file(wp,"https://huggingface.co/localoptima/paraketto/resolve/main/paraketto-fp16.bin");

    using clk=std::chrono::high_resolution_clock;
    auto t0=clk::now();
    Pipeline pipe;
    pipe.profile=profile;
    pipe.init(wp);
    auto t1=clk::now();
    fprintf(stderr,"startup: %.0fms\n",std::chrono::duration<double,std::milli>(t1-t0).count());

    for(auto& wf:wavs){
        WavData w=read_wav(wf);
        double dur=(double)w.samples.size()/w.sample_rate;
        auto ta=clk::now();
        std::string text=pipe.transcribe(w.samples.data(),w.samples.size());
        auto tb=clk::now();
        double el=std::chrono::duration<double>(tb-ta).count();
        printf("%s\n",text.c_str());
        fprintf(stderr,"%.1fs audio, %.1fms (mel=%.1f enc=%.1f dec=%.1f), %.0fx RTFx\n",
                dur,el*1000,pipe.last_mel_ms,pipe.last_enc_ms,pipe.last_dec_ms,dur/el);
    }
    return 0;
}
