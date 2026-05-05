#if defined(__ANDROID__) && !defined(NEON_GUI_SDL2)
#  define NEON_GUI_SDL2
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "imgui.h"
#include "implot.h"

#if defined(NEON_GUI_SDL2)
#  include "backends/imgui_impl_sdl2.h"
#  include "backends/imgui_impl_opengl3.h"
#  include <SDL.h>
#  if defined(__ANDROID__)
#    include <GLES2/gl2.h>
#    define IMGUI_IMPL_OPENGL_ES2
#  else
#    include <SDL_opengl.h>
#  endif
#else
#  include "backends/imgui_impl_glfw.h"
#  include "backends/imgui_impl_opengl3.h"
#  include <GLFW/glfw3.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define HAVE_NEON 1
   using i32x4 = int32x4_t;
   using i64x2 = int64x2_t;
#else
#  define HAVE_NEON 0
#endif

using i32 = int32_t;
using i64 = int64_t;
using Clock = std::chrono::high_resolution_clock;

static i64 process_array_scalar(const i32* __restrict__ data, size_t n)
{
    i64 sum = 0;
    for (size_t i = 0; i < n; ++i) {
        i32 val = data[i];
        if      (val > 0) sum += val;
        else if (val < 0) sum -= (i64)val;
    }
    return sum;
}

#if HAVE_NEON
static inline i32x4 neon_abs_barrel(i32x4 vec)
{
    i32x4 sign    = vshrq_n_s32(vec, 31);
    i32x4 abs_val = veorq_s32(vec, sign);
    abs_val        = vsubq_s32(abs_val, sign);
    return abs_val;
}

static i64 process_array_neon(const i32* __restrict__ data, size_t n)
{
    i64 sum = 0;
    size_t i = 0;
    i32x4 acc = vdupq_n_s32(0);
    if (n >= 4) __builtin_prefetch(data, 0, 3);
    for (; i + 3 < n; i += 4) {
        __builtin_prefetch(data + i + 8, 0, 1);
        i32x4 vec     = vld1q_s32(data + i);
        i32x4 abs_vec = neon_abs_barrel(vec);
        i32x4 mp      = vcgtq_s32(vec, vdupq_n_s32(0));
        i32x4 mn      = vcltq_s32(vec, vdupq_n_s32(0));
        i32x4 contrib = vorrq_s32(vandq_s32(vec, mp), vandq_s32(abs_vec, mn));
        acc = vaddq_s32(acc, contrib);
    }
#if defined(__aarch64__)
    sum = (i64)vaddlvq_s32(acc);
#else
    i64x2 a64 = vpaddlq_s32(acc);
    sum = vgetq_lane_s64(a64, 0) + vgetq_lane_s64(a64, 1);
#endif
    for (; i < n; ++i) {
        i32 val = data[i];
        if      (val > 0) sum += val;
        else if (val < 0) sum -= (i64)val;
    }
    return sum;
}

static i64 process_array_neon_unrolled(const i32* __restrict__ data, size_t n)
{
    i64 sum = 0;
    size_t i = 0;
    i32x4 acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
    i32x4 zero = vdupq_n_s32(0);
    for (; i + 7 < n; i += 8) {
        __builtin_prefetch(data + i + 16, 0, 1);
        i32x4 v0 = vld1q_s32(data + i);
        i32x4 v1 = vld1q_s32(data + i + 4);
        i32x4 s0 = vshrq_n_s32(v0, 31);
        i32x4 a0 = vsubq_s32(veorq_s32(v0, s0), s0);
        i32x4 s1 = vshrq_n_s32(v1, 31);
        i32x4 a1 = vsubq_s32(veorq_s32(v1, s1), s1);
        i32x4 nz0 = vorrq_s32(vcgtq_s32(v0, zero), vcltq_s32(v0, zero));
        i32x4 nz1 = vorrq_s32(vcgtq_s32(v1, zero), vcltq_s32(v1, zero));
        acc0 = vaddq_s32(acc0, vandq_s32(a0, nz0));
        acc1 = vaddq_s32(acc1, vandq_s32(a1, nz1));
    }
    i32x4 acc = vaddq_s32(acc0, acc1);
#if defined(__aarch64__)
    sum = (i64)vaddlvq_s32(acc);
#else
    i64x2 a64 = vpaddlq_s32(acc);
    sum = vgetq_lane_s64(a64, 0) + vgetq_lane_s64(a64, 1);
#endif
    for (; i < n; ++i) {
        i32 val = data[i];
        if      (val > 0) sum += val;
        else if (val < 0) sum -= (i64)val;
    }
    return sum;
}
#else
static i64 process_array_neon(const i32* d, size_t n)         { return process_array_scalar(d,n); }
static i64 process_array_neon_unrolled(const i32* d, size_t n){ return process_array_scalar(d,n); }
#endif

struct TestResult {
    std::string name;
    i64  scalar, neon, unrolled, expected;
    bool pass;
};

static std::vector<TestResult> run_correctness_tests()
{
    struct TC { const char* name; std::vector<i32> data; i64 expected; };
    std::vector<TC> cases = {
        {"Empty array",            {},                                    0},
        {"Single 0",               {0},                                   0},
        {"Single +42",             {42},                                 42},
        {"Single -7",              {-7},                                  7},
        {"All zeros",              {0,0,0,0,0},                          0},
        {"All positive",           {1,2,3,4,5},                         15},
        {"All negative",           {-1,-2,-3,-4,-5},                    15},
        {"Mixed",                  {-3,0,5,-2,0,1},                     11},
        {"Exactly 4 elements",     {-1,2,-3,4},                         10},
        {"5 elements (tail)",      {1,-1,1,-1,1},                        5},
        {"7 elements",             {3,-3,0,7,-7,0,1},                   21},
        {"INT32_MAX x2",           {INT32_MAX,INT32_MAX},  (i64)INT32_MAX*2},
        {"INT32_MIN+1",            {INT32_MIN+1},          (i64)INT32_MAX},
    };

    std::vector<TestResult> results;
    for (auto& t : cases) {
        TestResult r;
        r.name     = t.name;
        r.expected = t.expected;
        r.scalar   = process_array_scalar  (t.data.data(), t.data.size());
        r.neon     = process_array_neon    (t.data.data(), t.data.size());
        r.unrolled = process_array_neon_unrolled(t.data.data(), t.data.size());
        r.pass     = (r.scalar == r.expected)
                  && (r.neon   == r.expected)
                  && (r.unrolled == r.expected);
        results.push_back(r);
    }
    return results;
}

struct BenchRow {
    size_t  N;
    double  ms_scalar, ms_neon, ms_unrolled;
    double  speedup_neon, speedup_unrolled;
};

template<typename Fn>
static double bench_ms(Fn fn, int iters)
{
    volatile i64 dummy = 0;
    dummy = fn();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        dummy += fn();
    }
    auto t1 = Clock::now();
    (void)dummy;
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static std::vector<BenchRow> run_benchmark(int runs_per_size)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<i32> dist(-1000, 1000);

    std::vector<size_t> sizes = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000};
    std::vector<BenchRow> rows;
    
    for (size_t N : sizes) {
        double sum_scalar = 0, sum_neon = 0, sum_unrolled = 0;
        
        for (int run = 0; run < runs_per_size; ++run) {
            std::vector<i32> buf(N);
            for (auto& x : buf) x = dist(rng);
            for (size_t k = 0; k < N/5; ++k)
                buf[rng() % N] = 0;
            const i32* data = buf.data();
            
            sum_scalar  += bench_ms([&]{ return process_array_scalar(data, N); }, 10);
            sum_neon    += bench_ms([&]{ return process_array_neon(data, N); }, 10);
            sum_unrolled+= bench_ms([&]{ return process_array_neon_unrolled(data, N); }, 10);
        }
        
        BenchRow row;
        row.N          = N;
        row.ms_scalar  = sum_scalar / runs_per_size;
        row.ms_neon    = sum_neon / runs_per_size;
        row.ms_unrolled= sum_unrolled / runs_per_size;
        row.speedup_neon    = row.ms_scalar / row.ms_neon;
        row.speedup_unrolled= row.ms_scalar / row.ms_unrolled;
        rows.push_back(row);
    }
    return rows;
}

struct AppState {
    std::vector<TestResult> test_results;
    bool tests_done = false;

    std::vector<BenchRow>   bench_rows;
    bool  bench_running  = false;
    bool  bench_done     = false;
    int   bench_runs     = 10;
    std::thread bench_thread;
    std::mutex  bench_mtx;

    std::vector<double> plot_N;
    std::vector<double> plot_scalar, plot_neon, plot_unrolled;
    std::vector<double> plot_speedup_neon, plot_speedup_unrolled;

    int   custom_n     = 1000;
    int   seed         = 42;
    int   range        = 1000;
    float zero_frac    = 0.20f;
    i64   custom_scalar = 0, custom_neon = 0, custom_unrolled = 0;
    bool  custom_run   = false;

    void rebuild_plot_data() {
        std::lock_guard<std::mutex> lk(bench_mtx);
        plot_N.clear(); plot_scalar.clear(); plot_neon.clear(); plot_unrolled.clear();
        plot_speedup_neon.clear(); plot_speedup_unrolled.clear();
        for (auto& r : bench_rows) {
            plot_N.push_back((double)r.N);
            plot_scalar.push_back(r.ms_scalar);
            plot_neon.push_back(r.ms_neon);
            plot_unrolled.push_back(r.ms_unrolled);
            plot_speedup_neon.push_back(r.speedup_neon);
            plot_speedup_unrolled.push_back(r.speedup_unrolled);
        }
    }
};

static std::string fmt_n(size_t n) {
    std::ostringstream ss;
    if      (n >= 1000000000) ss << (n/1000000000) << "B";
    else if (n >= 1000000)    ss << (n/1000000) << "M";
    else if (n >= 1000)       ss << (n/1000) << "K";
    else                      ss << n;
    return ss.str();
}

static void apply_style(float dpi)
{
    ImGui::StyleColorsDark();
    ImGuiStyle& s = ImGui::GetStyle();
    s.ScaleAllSizes(dpi);
    s.WindowRounding    = 6.f * dpi;
    s.FrameRounding     = 4.f * dpi;
    s.GrabRounding      = 4.f * dpi;
    s.ScrollbarRounding = 6.f * dpi;
    s.FramePadding      = ImVec2(8.f*dpi, 5.f*dpi);
    s.ItemSpacing       = ImVec2(10.f*dpi, 6.f*dpi);

    ImVec4* c = s.Colors;
    c[ImGuiCol_Header]        = ImVec4(0.15f,0.55f,0.60f,0.65f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.20f,0.70f,0.75f,0.80f);
    c[ImGuiCol_Button]        = ImVec4(0.13f,0.50f,0.55f,0.90f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.18f,0.65f,0.70f,1.00f);
    c[ImGuiCol_ButtonActive]  = ImVec4(0.10f,0.40f,0.45f,1.00f);
    c[ImGuiCol_CheckMark]     = ImVec4(0.20f,0.85f,0.90f,1.00f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.10f,0.35f,0.40f,1.00f);
    c[ImGuiCol_Tab]           = ImVec4(0.10f,0.35f,0.40f,0.86f);
    c[ImGuiCol_TabHovered]    = ImVec4(0.20f,0.65f,0.70f,0.80f);
    c[ImGuiCol_TabActive]     = ImVec4(0.15f,0.55f,0.60f,1.00f);
}

static void draw_platform_panel(float dpi)
{
    ImGui::SetNextWindowPos (ImVec2(10*dpi,  10*dpi), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360*dpi,160*dpi), ImGuiCond_FirstUseEver);
    ImGui::Begin("Platform Info", nullptr, ImGuiWindowFlags_NoResize);

    ImGui::TextColored(ImVec4(0.20f,0.85f,0.90f,1.f), "ARM NEON Benchmark");
    ImGui::Separator(); ImGui::Spacing();

#if HAVE_NEON
    ImGui::TextColored(ImVec4(0.2f,1.0f,0.4f,1.f), "  NEON: AVAILABLE");
#  if defined(__aarch64__)
    ImGui::Text        ("  Mode : AArch64 (vaddlvq_s32)");
#  else
    ImGui::Text        ("  Mode : ARMv7 NEON (vpaddlq_s32)");
#  endif
#else
    ImGui::TextColored(ImVec4(1.0f,0.8f,0.2f,1.f), "  NEON: NOT AVAILABLE");
    ImGui::TextDisabled("  (scalar fallback used for all variants)");
#endif

    ImGui::Spacing();
    ImGui::TextDisabled("  Compiler: " __VERSION__);
    ImGui::TextDisabled("  Build date: " __DATE__);

    ImGui::End();
}

static void draw_tests_panel(float dpi, AppState& app)
{
    ImGui::SetNextWindowPos (ImVec2(10*dpi, 180*dpi), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(650*dpi,380*dpi), ImGuiCond_FirstUseEver);
    ImGui::Begin("Correctness Tests", nullptr);

    if (ImGui::Button("  Run Tests  ", ImVec2(160*dpi, 0))) {
        app.test_results = run_correctness_tests();
        app.tests_done   = true;
    }

    if (!app.tests_done) {
        ImGui::TextDisabled("Press 'Run Tests' to check all implementations.");
        ImGui::End(); return;
    }

    int passed = 0;
    for (auto& r : app.test_results) if (r.pass) ++passed;
    int total = (int)app.test_results.size();

    ImGui::SameLine(0, 20*dpi);
    if (passed == total)
        ImGui::TextColored(ImVec4(0.2f,1.0f,0.4f,1.f), "ALL %d/%d PASSED", passed, total);
    else
        ImGui::TextColored(ImVec4(1.0f,0.3f,0.3f,1.f), "%d/%d FAILED", total-passed, total);

    ImGui::Spacing();
    ImGui::Separator();

    ImGuiTableFlags tflags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    ImVec2 tsize(0.f, 280*dpi);

    if (ImGui::BeginTable("##tests", 6, tflags, tsize)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Test",      ImGuiTableColumnFlags_WidthStretch, 2.5f);
        ImGui::TableSetupColumn("Expected",  ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Scalar",    ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("NEON",      ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Unrolled",  ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Status",    ImGuiTableColumnFlags_WidthFixed,  70*dpi);
        ImGui::TableHeadersRow();

        for (auto& r : app.test_results) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.name.c_str());

            ImGui::TableNextColumn(); ImGui::Text("%lld", (long long)r.expected);
            ImGui::TableNextColumn(); ImGui::Text("%lld", (long long)r.scalar);
            ImGui::TableNextColumn(); ImGui::Text("%lld", (long long)r.neon);
            ImGui::TableNextColumn(); ImGui::Text("%lld", (long long)r.unrolled);

            ImGui::TableNextColumn();
            if (r.pass)
                ImGui::TextColored(ImVec4(0.2f,1.0f,0.4f,1.f), "PASS");
            else
                ImGui::TextColored(ImVec4(1.0f,0.3f,0.3f,1.f), "FAIL");
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

static void draw_custom_panel(float dpi, AppState& app)
{
    ImGui::SetNextWindowPos (ImVec2(670*dpi, 10*dpi), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360*dpi,300*dpi), ImGuiCond_FirstUseEver);
    ImGui::Begin("Custom Array", nullptr);

    ImGui::TextColored(ImVec4(0.20f,0.85f,0.90f,1.f), "Generate & process");
    ImGui::Separator(); ImGui::Spacing();

    ImGui::SetNextItemWidth(160*dpi);
    ImGui::InputInt("N elements", &app.custom_n);
    app.custom_n = std::max(1, std::min(app.custom_n, 10000000));

    ImGui::SetNextItemWidth(160*dpi);
    ImGui::InputInt("RNG seed", &app.seed);

    ImGui::SetNextItemWidth(160*dpi);
    ImGui::InputInt("Value range ±", &app.range);
    app.range = std::max(1, app.range);

    ImGui::SetNextItemWidth(160*dpi);
    ImGui::SliderFloat("Zero fraction", &app.zero_frac, 0.0f, 1.0f, "%.2f");

    ImGui::Spacing();
    if (ImGui::Button("  Process  ", ImVec2(120*dpi, 0))) {
        std::mt19937 rng((unsigned)app.seed);
        std::uniform_int_distribution<i32> dist(-app.range, app.range);
        std::vector<i32> buf(app.custom_n);
        for (auto& x : buf) x = dist(rng);
        size_t zeros = (size_t)(app.custom_n * app.zero_frac);
        for (size_t k = 0; k < zeros; ++k) buf[rng() % app.custom_n] = 0;

        app.custom_scalar   = process_array_scalar  (buf.data(), buf.size());
        app.custom_neon     = process_array_neon    (buf.data(), buf.size());
        app.custom_unrolled = process_array_neon_unrolled(buf.data(), buf.size());
        app.custom_run      = true;
    }

    if (app.custom_run) {
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        bool ok = (app.custom_scalar == app.custom_neon)
               && (app.custom_scalar == app.custom_unrolled);
        auto row = [&](const char* lbl, i64 val){
            ImGui::TextDisabled("  %-12s", lbl);
            ImGui::SameLine(150*dpi);
            ImGui::Text("%lld", (long long)val);
        };
        row("Scalar :",   app.custom_scalar);
        row("NEON :",     app.custom_neon);
        row("Unrolled :", app.custom_unrolled);
        ImGui::Spacing();
        if (ok)
            ImGui::TextColored(ImVec4(0.2f,1.0f,0.4f,1.f), "  Results match");
        else
            ImGui::TextColored(ImVec4(1.0f,0.3f,0.3f,1.f), "  MISMATCH!");
    }
    ImGui::End();
}

static void draw_bench_control_panel(float dpi, AppState& app)
{
    ImGui::SetNextWindowPos (ImVec2(670*dpi, 320*dpi), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360*dpi,240*dpi), ImGuiCond_FirstUseEver);
    ImGui::Begin("Benchmark Control", nullptr);

    ImGui::TextColored(ImVec4(0.20f,0.85f,0.90f,1.f), "Performance Benchmark");
    ImGui::Separator(); ImGui::Spacing();

    ImGui::SetNextItemWidth(160*dpi);
    ImGui::InputInt("Runs per size", &app.bench_runs);
    app.bench_runs = std::max(1, std::min(app.bench_runs, 50));

    ImGui::Spacing();

    if (!app.bench_running) {
        if (ImGui::Button("  Run Benchmark  ", ImVec2(-1, 0))) {
            app.bench_running = true;
            app.bench_done    = false;
            int runs = app.bench_runs;
            app.bench_thread = std::thread([&app, runs](){
                auto rows = run_benchmark(runs);
                {
                    std::lock_guard<std::mutex> lk(app.bench_mtx);
                    app.bench_rows = std::move(rows);
                }
                app.bench_done    = true;
                app.bench_running = false;
                app.rebuild_plot_data();
            });
            app.bench_thread.detach();
        }
    } else {
        ImGui::TextColored(ImVec4(1.0f,0.8f,0.2f,1.f), "  Running...");
        float t = (float)ImGui::GetTime();
        const char* dots[] = {"   ", ".  ", ".. ", "..."};
        ImGui::SameLine();
        ImGui::Text("%s", dots[(int)(t*2) & 3]);
    }

    if (app.bench_done) {
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        std::lock_guard<std::mutex> lk(app.bench_mtx);
        for (auto& r : app.bench_rows) {
            double best_speedup = std::max(r.speedup_neon, r.speedup_unrolled);
            ImVec4 col = (best_speedup > 1.5) ? ImVec4(0.2f,1.0f,0.4f,1.f) : ImVec4(1.0f,0.8f,0.2f,1.f);
            ImGui::TextColored(col, "  N=%-7s  best x%.2f", fmt_n(r.N).c_str(), best_speedup);
        }
    }

    ImGui::End();
}

static void draw_bench_table(float dpi, AppState& app)
{
    if (!app.bench_done) return;

    ImGui::SetNextWindowPos (ImVec2(10*dpi, 570*dpi), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(650*dpi,220*dpi), ImGuiCond_FirstUseEver);
    ImGui::Begin("Benchmark Results", nullptr);

    ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg
                       | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("##bench", 7, tf)) {
        ImGui::TableSetupColumn("N",         ImGuiTableColumnFlags_WidthFixed,  70*dpi);
        ImGui::TableSetupColumn("Scalar ms", ImGuiTableColumnFlags_WidthStretch,1.5f);
        ImGui::TableSetupColumn("NEON ms",   ImGuiTableColumnFlags_WidthStretch,1.5f);
        ImGui::TableSetupColumn("Unrolled ms",ImGuiTableColumnFlags_WidthStretch,1.5f);
        ImGui::TableSetupColumn("NEON x",    ImGuiTableColumnFlags_WidthStretch,1.f);
        ImGui::TableSetupColumn("Unroll x",  ImGuiTableColumnFlags_WidthStretch,1.f);
        ImGui::TableSetupColumn("Best",      ImGuiTableColumnFlags_WidthFixed,  80*dpi);
        ImGui::TableHeadersRow();

        std::lock_guard<std::mutex> lk(app.bench_mtx);
        for (auto& r : app.bench_rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%s", fmt_n(r.N).c_str());
            ImGui::TableNextColumn(); ImGui::Text("%.4f", r.ms_scalar);
            ImGui::TableNextColumn(); ImGui::Text("%.4f", r.ms_neon);
            ImGui::TableNextColumn(); ImGui::Text("%.4f", r.ms_unrolled);

            auto speedup_color = [](double s) -> ImVec4 {
                return (s > 2.0) ? ImVec4(0.2f,1.0f,0.4f,1.f)
                     : (s > 1.0) ? ImVec4(1.0f,0.9f,0.2f,1.f)
                                 : ImVec4(1.0f,0.4f,0.4f,1.f);
            };

            ImGui::TableNextColumn();
            ImGui::TextColored(speedup_color(r.speedup_neon), "x%.2f", r.speedup_neon);
            ImGui::TableNextColumn();
            ImGui::TextColored(speedup_color(r.speedup_unrolled), "x%.2f", r.speedup_unrolled);

            ImGui::TableNextColumn();
            bool neon_wins = r.speedup_neon > r.speedup_unrolled;
            ImGui::TextColored(ImVec4(0.20f,0.85f,0.90f,1.f),
                               "%s", neon_wins ? "NEON" : "UNROLL");
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

static void draw_plots(float dpi, AppState& app)
{
    if (!app.bench_done) return;
    if (app.plot_N.empty()) return;

    const int n = (int)app.plot_N.size();
    const double* xs = app.plot_N.data();

    ImGui::SetNextWindowPos (ImVec2(10*dpi, 800*dpi), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(650*dpi,340*dpi), ImGuiCond_FirstUseEver);
    ImGui::Begin("Execution Time (ms)", nullptr);

    if (ImPlot::BeginPlot("##time", ImVec2(-1,-1))) {
        ImPlot::SetupAxes("Array size (N)", "Time (ms)");
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        ImPlot::SetupLegend(ImPlotLocation_NorthWest);

        ImPlot::PlotLine("Scalar",   xs, app.plot_scalar.data(),   n);
        ImPlot::PlotLine("NEON",     xs, app.plot_neon.data(),     n);
        ImPlot::PlotLine("Unrolled", xs, app.plot_unrolled.data(), n);

        ImPlot::PlotScatter("##sm", xs, app.plot_scalar.data(), n);
        ImPlot::PlotScatter("##nm", xs, app.plot_neon.data(), n);
        ImPlot::PlotScatter("##um", xs, app.plot_unrolled.data(), n);

        ImPlot::EndPlot();
    }
    ImGui::End();

    ImGui::SetNextWindowPos (ImVec2(670*dpi, 570*dpi), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360*dpi,340*dpi), ImGuiCond_FirstUseEver);
    ImGui::Begin("Speedup vs Scalar", nullptr);

    if (ImPlot::BeginPlot("##speedup", ImVec2(-1,-1))) {
        ImPlot::SetupAxes("Array size (N)", "Speedup (x)");
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupLegend(ImPlotLocation_NorthWest);

        {
            std::vector<double> baseline_xs = {app.plot_N.front(), app.plot_N.back()};
            std::vector<double> baseline_ys = {1.0, 1.0};
            ImPlot::PlotLine("Baseline x1", baseline_xs.data(), baseline_ys.data(), 2);
        }

        ImPlot::PlotLine("NEON",     xs, app.plot_speedup_neon.data(),     n);
        ImPlot::PlotLine("Unrolled", xs, app.plot_speedup_unrolled.data(), n);

        ImPlot::EndPlot();
    }
    ImGui::End();

    {
        auto& last = app.bench_rows.back();
        ImGui::SetNextWindowPos (ImVec2(670*dpi, 920*dpi), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360*dpi,220*dpi), ImGuiCond_FirstUseEver);
        ImGui::Begin("Time at N=1B (bar)", nullptr);

        if (ImPlot::BeginPlot("##bar1b", ImVec2(-1,-1))) {
            ImPlot::SetupAxes("Implementation", "Time (ms)",
                              ImPlotAxisFlags_NoTickLabels, 0);
            
            double vals[3] = {last.ms_scalar, last.ms_neon, last.ms_unrolled};
            const char* lbls[3] = {"Scalar","NEON","Unrolled"};
            
            ImPlot::PlotBarGroups(lbls, &vals[0], 3, 1, 0.67, 0);

            ImPlot::EndPlot();
        }
        ImGui::End();
    }
}
#if defined(NEON_GUI_SDL2)

struct WindowContext { SDL_Window* window; SDL_GLContext gl_ctx; };

static bool window_init(WindowContext& ctx, float& dpi, int w, int h)
{
    SDL_Init(SDL_INIT_VIDEO);
#if defined(__ANDROID__)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    ctx.window = SDL_CreateWindow("NEON Benchmark",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        0, 0,
        SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN | SDL_WINDOW_ALLOW_HIGHDPI);
    dpi = 2.0f;
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    ctx.window = SDL_CreateWindow("NEON Benchmark",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    dpi = 1.0f;
#endif
    if (!ctx.window) return false;
    ctx.gl_ctx = SDL_GL_CreateContext(ctx.window);
    SDL_GL_MakeCurrent(ctx.window, ctx.gl_ctx);
    SDL_GL_SetSwapInterval(1);
    return true;
}

static bool window_should_close(WindowContext& ctx)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        ImGui_ImplSDL2_ProcessEvent(&e);
        if (e.type == SDL_QUIT) return true;
        if (e.type == SDL_WINDOWEVENT
            && e.window.event == SDL_WINDOWEVENT_CLOSE
            && e.window.windowID == SDL_GetWindowID(ctx.window)) return true;
    }
    return false;
}

static void window_new_frame(WindowContext& ctx)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
}

static void window_present(WindowContext& ctx)
{
    int w, h;
    SDL_GetWindowSize(ctx.window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.10f,0.12f,0.14f,1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(ctx.window);
}

static void window_shutdown(WindowContext& ctx)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    SDL_GL_DeleteContext(ctx.gl_ctx);
    SDL_DestroyWindow(ctx.window);
    SDL_Quit();
}

static void imgui_backend_init(WindowContext& ctx)
{
    ImGui_ImplSDL2_InitForOpenGL(ctx.window, ctx.gl_ctx);
#if defined(__ANDROID__)
    ImGui_ImplOpenGL3_Init("#version 100");
#else
    ImGui_ImplOpenGL3_Init("#version 330");
#endif
}

#else

struct WindowContext { GLFWwindow* window; };

static bool window_init(WindowContext& ctx, float& dpi, int w, int h)
{
    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    dpi = 1.0f;
    GLFWmonitor* mon = glfwGetPrimaryMonitor();
    if (mon) {
        float xs, ys;
        glfwGetMonitorContentScale(mon, &xs, &ys);
        dpi = std::max(xs, ys);
        if (dpi < 1.0f) dpi = 1.0f;
    }

    ctx.window = glfwCreateWindow((int)(w*dpi),(int)(h*dpi),
                                  "NEON Benchmark", nullptr, nullptr);
    if (!ctx.window) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(ctx.window);
    glfwSwapInterval(1);
    return true;
}

static bool window_should_close(WindowContext& ctx)
{
    glfwPollEvents();
    return glfwWindowShouldClose(ctx.window);
}

static void window_new_frame(WindowContext&)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

static void window_present(WindowContext& ctx)
{
    int w, h;
    glfwGetFramebufferSize(ctx.window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.10f,0.12f,0.14f,1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(ctx.window);
}

static void window_shutdown(WindowContext& ctx)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    glfwDestroyWindow(ctx.window);
    glfwTerminate();
}

static void imgui_backend_init(WindowContext& ctx)
{
    ImGui_ImplGlfw_InitForOpenGL(ctx.window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

#endif

int main()
{
    WindowContext ctx{};
    float dpi = 1.0f;

    if (!window_init(ctx, dpi, 1060, 1160)) return 1;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.FontGlobalScale = dpi;

    apply_style(dpi);
    imgui_backend_init(ctx);

    AppState app;

    while (!window_should_close(ctx)) {
        window_new_frame(ctx);

        draw_platform_panel    (dpi);
        draw_tests_panel       (dpi, app);
        draw_custom_panel      (dpi, app);
        draw_bench_control_panel(dpi, app);
        draw_bench_table       (dpi, app);
        draw_plots             (dpi, app);

        ImGui::Render();
        window_present(ctx);
    }

    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    window_shutdown(ctx);
    return 0;
}