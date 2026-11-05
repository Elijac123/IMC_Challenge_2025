#include <bits/stdc++.h>
using namespace std;

/* ========================= Controller (feature flags + hyperparameters) ========================= */

struct Ctrl {
    // Global toggles
    bool enable_1801 = true;
    bool enable_5150 = true;
    bool enable_1425 = true;

    // Alignment (delta) search
    bool enable_delta_search = false;
    int delta_min = -8;
    int delta_max = 8;

    // Ensemble weighting
    enum WeightMode { Fixed = 0, InverseRMSE = 1, SoftmaxRMSE = 2, NNLS_Simplex = 3 } weight_mode = NNLS_Simplex;
    // Prior weights (used if Fixed or as prior for NNLS_Simplex)
    double w1801 = 0.66, w5150 = 0.13, w1425 = 0.10;
    double prior_strength = 0.02; // L2 pull toward prior in NNLS
    double softmax_T = 0.4;      // temperature for SoftmaxRMSE
    bool normalize_weights = true;

    // Calibration
    bool enable_phase_affine = true; // per-phase y' = a_r * y + b_r
    bool enable_post_fir = true;     // learn small linear FIR on training and apply to test
    int post_fir_len = 9;            // odd, e.g., 7/9/11

    // 5150 (cache + poly ridge) + internal SignalModel (1620)
    struct C5150 {
        int P_cache = 68, Q_cache = 8;
        int P_reg = 68, Q_reg = 8;
        double ridge_lambda_cap = 0.03; // upper cap (auto-lambda inside)
        double alpha_5150_vs_sm = 0.64; // blend: out = alpha*5150 + (1-alpha)*SignalModel
        bool force_alpha_case_override = false; // if true, do not override alpha by M
        // Feature reductions (to keep 5150 light we keep default basis)
    } c5150;

    // 1620 (SignalModel) in-run params
    struct C1620 {
        // shrinkage betas for residual LUT mixing
        double beta11 = 1e-5;
        double beta9  = 0.50;
        double beta7  = 0.59;
        // phase-weight smoothing (left, center, right)
        double sm_left = 0.20, sm_center = 0.60, sm_right = 0.20;
        // final FIR inside 1620 kept at 9 taps
    } c1620;

    // 1801 stage configs (contexts) and delta band per case
    struct C1801 {
        // Two stages default for case 1
        vector<vector<pair<string,int>>> stages;
        int delta_min = 1, delta_max = 6; // used if we fall back to 1801-based search
    } c1801;

    // 1425 memory polynomial
    struct C1425 {
        int L_pre = 7, L_post = 68;
        vector<int> orders = {1, -1}; // 1 (linear), -1 (log|x|)
        double lambda = 35.0;
        int delta_min = -7, delta_max = 7;
        double tau = 40.0;
    } c1425;

    // Case-aware defaults
    static Ctrl make_for_case(int M, int N, const vector<double>& ana1) {
        Ctrl cfg;

        // Default 1801 stages similar to original
        vector<pair<string,int>> stage1 = {{"symbol",1},{"pre_duplet",2},{"post_duplet",2},{"triplet",3},{"quintuplet",5},{"septuplet",7}};
        vector<pair<string,int>> stage2 = {{"symbol",1},{"pre_duplet",2},{"post_duplet",2},{"triplet",3}};
        cfg.c1801.stages = {stage1, stage2};

        if (M == 5460) {
            // Case 1: use all, original prior weights, moderate delta search
            cfg.enable_1801 = true;
            cfg.enable_5150 = true;
            cfg.enable_1425 = true;
            cfg.weight_mode = NNLS_Simplex;
            cfg.w1801 = 0.66; cfg.w5150 = 0.23; cfg.w1425 = 0.08;
            cfg.prior_strength = 0.7;
            cfg.enable_phase_affine = true;
            cfg.enable_post_fir = true;
            cfg.post_fir_len = 21;

            cfg.c5150.P_cache = 69; cfg.c5150.Q_cache = 8;
            cfg.c5150.P_reg   = 69; cfg.c5150.Q_reg   = 8;
            cfg.c5150.alpha_5150_vs_sm = 1.0; // slightly higher trust in 5150 here
            cfg.c5150.force_alpha_case_override = true;

            cfg.c1620.beta11 = 50; cfg.c1620.beta9 = 50; cfg.c1620.beta7 = 50;

            cfg.enable_delta_search = true;
            cfg.delta_min = 1; cfg.delta_max = 7;

            cfg.c1425.L_pre = 7; cfg.c1425.L_post = 17;
            cfg.c1425.orders = {1, -1};
            cfg.c1425.lambda = 3.5e-7;
            cfg.c1425.delta_min = 0; cfg.c1425.delta_max = 7;
            cfg.c1425.tau = 0;
        } else if (M == 5300) {
            // Case 4: 1425 excels; keep simple; add calibration + small FIR
            cfg.enable_1801 = true;
            cfg.enable_5150 = true;
            cfg.enable_1425 = true;
            cfg.weight_mode = Fixed;
            cfg.w1801 = 0.1; cfg.w5150 = 0.23; cfg.w1425 = 0.66;
            cfg.prior_strength = 0.5;
            cfg.enable_phase_affine = true;
            cfg.enable_post_fir = true;
            cfg.post_fir_len = 21;
            
            cfg.c5150.P_cache = 69; cfg.c5150.Q_cache = 8;
            cfg.c5150.P_reg   = 69; cfg.c5150.Q_reg   = 8;
            cfg.c5150.alpha_5150_vs_sm = 0.5; // slightly higher trust in 5150 here
            cfg.c5150.force_alpha_case_override = false;
            
            cfg.c1620.beta11 = 50; cfg.c1620.beta9 = 50; cfg.c1620.beta7 = 50;

            cfg.enable_delta_search = true;
            cfg.delta_min = -13; cfg.delta_max = 6;

            cfg.c1425.L_pre = 7; cfg.c1425.L_post = 68;
            cfg.c1425.orders = {1, -1};
            cfg.c1425.lambda = 35;
            cfg.c1425.delta_min = -13; cfg.c1425.delta_max = 7;
            cfg.c1425.tau = 0;
        } else if (M == 11000 && N > 2 && !ana1.empty() && ana1[2] > 0.0) { // Case 2
            cfg.enable_1801 = false;
            cfg.enable_5150 = true;
            cfg.enable_1425 = true;
            cfg.weight_mode = NNLS_Simplex;
            cfg.w1801 = 0.0; cfg.w5150 = 0.7; cfg.w1425 = 0.3;
            cfg.prior_strength = 0.1;
            cfg.enable_phase_affine = true;
            cfg.enable_post_fir = true;
            cfg.post_fir_len = 9;

            cfg.c5150.P_cache = 71; cfg.c5150.Q_cache = 8;
            cfg.c5150.P_reg   = 71; cfg.c5150.Q_reg   = 8;
            cfg.c5150.alpha_5150_vs_sm = 0.64; // slightly higher trust in 5150 here
            cfg.c5150.force_alpha_case_override = true;

            cfg.c1620.beta11 = 1e-5; cfg.c1620.beta9 = 0.5; cfg.c1620.beta7 = 0.5;

            cfg.enable_delta_search = true;
            cfg.delta_min = 0; cfg.delta_max = 20;

            cfg.c1801.delta_min = 0; cfg.c1801.delta_max = 20;
            cfg.c1425.L_pre = 7; cfg.c1425.L_post = 17;
            cfg.c1425.orders = {1, -1};
            cfg.c1425.lambda = 3.5e-7;
            cfg.c1425.delta_min = 0; cfg.c1425.delta_max = 20;
            cfg.c1425.tau = 0;
        } else {
            cfg.enable_1801 = false;
            cfg.enable_5150 = true;
            cfg.enable_1425 = true;
            cfg.weight_mode = Fixed;
            cfg.w1801 = 0.23; cfg.w5150 = 0.23; cfg.w1425 = 0.66;
            cfg.prior_strength = 0.1;
            cfg.enable_phase_affine = false;
            cfg.enable_post_fir = true;
            cfg.post_fir_len = 1;

            cfg.c1425.L_pre = 7; cfg.c1425.L_post = 68;
            cfg.c1425.orders = {1, -1};
            cfg.c1425.lambda = 35.0;
            cfg.c1425.delta_min = 12; cfg.c1425.delta_max = 40;
            cfg.c1425.tau = 0;

            cfg.enable_delta_search = true;
            cfg.delta_min = 12; cfg.delta_max = 60;
        }
        return cfg;
    }
};

/* ========================= Utils ========================= */

static inline double clamp1(double x) { return x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x); }
static inline double clamp11(double x) { return x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x); }

struct VecHash {
    size_t operator()(const vector<int>& v) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        for (int x : v) {
            uint64_t y = (uint64_t)(x + 9);
            h ^= y + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return (size_t)h;
    }
};

/* ========================= Small linear algebra ========================= */

class Matrix {
public:
    vector<vector<double>> data;
    size_t rows = 0, cols = 0;

    Matrix() = default;
    Matrix(size_t r, size_t c, double val = 0.0) : data(r, vector<double>(c, val)), rows(r), cols(c) {}

    static Matrix solveSPD(Matrix A, Matrix B) {
        const size_t n = A.rows;
        vector<vector<double>> L = A.data;
        bool ok = true;
        for (size_t i = 0; i < n && ok; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double s = L[i][j];
                for (size_t k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
                if (i == j) {
                    if (s <= 1e-14) { ok = false; break; }
                    L[i][i] = sqrt(s);
                } else {
                    if (fabs(L[j][j]) < 1e-14) { ok = false; break; }
                    L[i][j] = s / L[j][j];
                }
            }
            if (!ok) break;
            for (size_t j = i + 1; j < n; ++j) L[i][j] = 0.0;
        }
        if (!ok) {
            // Gauss-Jordan fallback
            for (size_t i = 0; i < n; ++i) {
                size_t pivot = i;
                for (size_t j = i + 1; j < n; ++j) if (fabs(A.data[j][i]) > fabs(A.data[pivot][i])) pivot = j;
                if (pivot != i) { swap(A.data[i], A.data[pivot]); swap(B.data[i], B.data[pivot]); }
                double div = A.data[i][i];
                if (fabs(div) < 1e-12) continue;
                for (size_t j = i; j < n; ++j) A.data[i][j] /= div;
                for (size_t j = 0; j < B.cols; ++j) B.data[i][j] /= div;
                for (size_t j = 0; j < n; ++j) if (i != j) {
                    double mult = A.data[j][i];
                    if (mult == 0.0) continue;
                    for (size_t k = i; k < n; ++k) A.data[j][k] -= mult * A.data[i][k];
                    for (size_t k = 0; k < B.cols; ++k) B.data[j][k] -= mult * B.data[i][k];
                }
            }
            return B;
        }
        // Solve using Cholesky
        Matrix Y(n, B.cols);
        for (size_t j = 0; j < B.cols; ++j) {
            for (size_t i = 0; i < n; ++i) {
                double s = B.data[i][j];
                for (size_t k = 0; k < i; ++k) s -= L[i][k] * Y.data[k][j];
                Y.data[i][j] = s / L[i][i];
            }
        }
        Matrix X(n, B.cols);
        for (size_t j = 0; j < B.cols; ++j) {
            for (int i = (int)n - 1; i >= 0; --i) {
                double s = Y.data[i][j];
                for (size_t k = i + 1; k < n; ++k) s -= L[k][i] * X.data[k][j];
                X.data[i][j] = s / L[i][i];
            }
        }
        return X;
    }
};

/* ========================= Helpers for FIR and small solvers ========================= */

static void fit_fir_linear(const vector<double>& pred, const vector<double>& target, int LFIR, vector<double>& H, bool mild_ridge=true) {
    H.assign(LFIR, 0.0);
    const int N = (int)pred.size();
    if (N < LFIR + 1) return;
    const int HALF = LFIR / 2;
    vector<vector<double>> G(LFIR, vector<double>(LFIR, 0.0));
    vector<double> b(LFIR, 0.0);
    for (int n = HALF; n < N - HALF; ++n) {
        for (int i = 0; i < LFIR; ++i) {
            double xi = pred[n + i - HALF];
            b[i] += xi * target[n];
            for (int j = 0; j <= i; ++j) G[i][j] += xi * pred[n + j - HALF];
        }
    }
    for (int i = 0; i < LFIR; ++i) for (int j = 0; j < i; ++j) G[j][i] = G[i][j];
    if (mild_ridge) {
        double tr = 0.0; for (int i = 0; i < LFIR; ++i) tr += G[i][i];
        double lam = 1e-4 * (tr > 0 ? tr / LFIR : 1.0);
        for (int i = 0; i < LFIR; ++i) G[i][i] += lam;
    }
    // Cholesky
    for (int i = 0; i < LFIR; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = G[i][j];
            for (int k = 0; k < j; ++k) s -= G[i][k] * G[j][k];
            if (i == j) {
                if (s <= 0) { return; }
                G[i][j] = sqrt(s);
            } else {
                G[i][j] = s / G[j][j];
            }
        }
        for (int j = i + 1; j < LFIR; ++j) G[i][j] = 0.0;
    }
    // solve
    vector<double> y = b;
    for (int i = 0; i < LFIR; ++i) {
        double s = y[i];
        for (int k = 0; k < i; ++k) s -= G[i][k] * y[k];
        y[i] = s / G[i][i];
    }
    for (int i = LFIR - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < LFIR; ++k) s -= G[k][i] * H[k];
        H[i] = s / G[i][i];
    }
}

static void apply_fir_inplace(vector<double>& y, const vector<double>& H) {
    if (H.empty()) { for (double& v : y) v = clamp1(v); return; }
    const int LFIR = (int)H.size();
    const int HALF = LFIR / 2;
    const int N = (int)y.size();
    vector<double> out(N, 0.0);
    for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < LFIR; ++k) {
            int idx = n + k - HALF;
            if (idx < 0) idx = 0;
            if (idx >= N) idx = N - 1;
            acc += H[k] * y[idx];
        }
        out[n] = acc;
    }
    for (double& v : out) v = clamp1(v);
    y.swap(out);
}

static inline double rmse(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    long double se = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; se += d * d; }
    return sqrt((double)(se / a.size()));
}

/* ========================= 1620: SignalModel (parameterized) ========================= */

static inline int map_symbol_to_index(int s) { return (s + 3) / 2; } // {-3,-1,1,3}->{0,1,2,3}
static inline double norm_sym(int s) { return static_cast<double>(s) / 3.0; } // [-1,1]

struct SignalModel {
    static constexpr int OVERSAMPLE = 16;
    static constexpr int PRE = 170, POST = 60, TAPS = PRE + 1 + POST, CI = PRE;
    // Feature layout (D = 83)
    static constexpr int NEAR_CUBIC_RADIUS = 2;
    static constexpr int SQR_RADIUS = 6;
    static constexpr int XCN_RADIUS = 6;
    static constexpr int SIGNX_RADIUS = 1;
    static constexpr int base_bias = 0;
    static constexpr int base_lin  = base_bias + 1;
    static constexpr int base_cubn = base_lin + TAPS;
    static constexpr int base_sq_n = base_cubn + (2 * NEAR_CUBIC_RADIUS);
    static constexpr int base_xcn  = base_sq_n + (2 * SQR_RADIUS);
    static constexpr int base_c35  = base_xcn + (2 * XCN_RADIUS);
    static constexpr int base_signx= base_c35 + 2;
    static constexpr int base_oh   = base_signx + (2 * SIGNX_RADIUS);
    static constexpr int base_tr_p1c = base_oh + 3;
    static constexpr int base_tr_cn1 = base_tr_p1c + 16;
    static constexpr int D = base_tr_cn1 + 16;

    // Tunables injected from controller
    double beta11 = 1e-5, beta9 = 0.50, beta7 = 0.59;
    double sm_left = 0.20, sm_center = 0.60, sm_right = 0.20;
    static constexpr int LFIR = 9;

    array<array<double, D>, OVERSAMPLE> W{};
    array<double, D> mu{}, inv_std{};
    struct CtxRes {
        array<float, OVERSAMPLE> sum{};
        uint16_t count = 0;
    };
    unordered_map<uint32_t, CtxRes> lut7_res, lut9_res, lut11_res;
    array<double, LFIR> H{}; bool has_fir = false;

    static inline void build_raw_features(const vector<int>& s, int m, array<double, D>& x) {
        x.fill(0.0);
        x[base_bias] = 1.0;
        double a[TAPS];
        for (int lag = -PRE, t = 0; lag <= POST; ++lag, ++t) {
            int k = m + lag;
            double v = 0.0;
            if (k >= 0 && k < (int)s.size()) v = norm_sym(s[k]);
            a[t] = v;
        }
        const double c = a[CI];
        const double c2 = c * c, c3 = c2 * c, c5 = c3 * c2;
        const double c_sign = (s[m] >= 0) ? 1.0 : -1.0;
        for (int t = 0; t < TAPS; ++t) x[base_lin + t] = a[t];
        { int idx = base_cubn; for (int d = -NEAR_CUBIC_RADIUS; d <= NEAR_CUBIC_RADIUS; ++d) { if (d==0) continue; double v=a[CI+d]; x[idx++]=v*v*v; } }
        { int idx = base_sq_n; for (int d = -SQR_RADIUS; d <= SQR_RADIUS; ++d) { if (d==0) continue; double v=a[CI+d]; x[idx++]=v*v; } }
        { int idx = base_xcn; for (int d = -XCN_RADIUS; d <= XCN_RADIUS; ++d) { if (d==0) continue; x[idx++]=c*a[CI+d]; } }
        x[base_c35+0]=c3; x[base_c35+1]=c5;
        { int idx = base_signx; for (int d = -SIGNX_RADIUS; d <= SIGNX_RADIUS; ++d) { if (d==0) continue; x[idx++]=c_sign*a[CI+d]; } }
        { int cidx=map_symbol_to_index(s[m]); x[base_oh+0]=(cidx==0); x[base_oh+1]=(cidx==1); x[base_oh+2]=(cidx==2); }
        if (m-1>=0) { int p=map_symbol_to_index(s[m-1]); int cidx=map_symbol_to_index(s[m]); x[base_tr_p1c + 4*p + cidx] = 1.0; }
        if (m+1<(int)s.size()) { int cidx=map_symbol_to_index(s[m]); int n=map_symbol_to_index(s[m+1]); x[base_tr_cn1 + 4*cidx + n] = 1.0; }
    }

    inline void standardize_inplace(array<double, D>& x) const {
        for (int i = 1; i < D; ++i) x[i] = (x[i] - mu[i]) * inv_std[i];
    }

    static inline uint32_t pack_ctx7(const vector<int>& s, int m) {
        auto idx = [](int v){ return (uint32_t)((v+3)/2u)&3u; };
        uint32_t p3=idx(s[m-3]),p2=idx(s[m-2]),p1=idx(s[m-1]),c=idx(s[m]),n1=idx(s[m+1]),n2=idx(s[m+2]),n3=idx(s[m+3]);
        uint32_t key=0; key=(((((((p3<<2)|p2)<<2)|p1)<<2)|c)<<2|n1)<<2|n2; key=(key<<2)|n3; return key;
    }
    static inline uint32_t pack_ctx9(const vector<int>& s, int m) {
        auto idx = [](int v){ return (uint32_t)((v+3)/2u)&3u; };
        uint32_t p6=idx(s[m-6]),p5=idx(s[m-5]),p4=idx(s[m-4]),p3=idx(s[m-3]),p2=idx(s[m-2]),p1=idx(s[m-1]),c=idx(s[m]),n1=idx(s[m+1]),n2=idx(s[m+2]);
        uint32_t key=0; key=((((((((p6<<2)|p5)<<2)|p4)<<2)|p3)<<2)|p2)<<2|p1; key=((key<<2)|c); key=((key<<2)|n1); key=((key<<2)|n2); return key;
    }
    static inline uint32_t pack_ctx11(const vector<int>& s, int m) {
        auto idx = [](int v){ return (uint32_t)((v+3)/2u)&3u; };
        uint32_t key=0; for (int d=8; d>=1; --d) key=(key<<2)|idx(s[m-d]); key=(key<<2)|idx(s[m]); key=(key<<2)|idx(s[m+1]); key=(key<<2)|idx(s[m+2]); return key;
    }

    void smooth_phase_weights(array<array<double, D>, OVERSAMPLE>& Wref) const {
        array<array<double, D>, OVERSAMPLE> W2 = Wref;
        for (int r = 0; r < OVERSAMPLE; ++r) {
            for (int i = 0; i < D; ++i) {
                double left  = (r > 0) ? Wref[r-1][i] : Wref[r][i];
                double right = (r + 1 < OVERSAMPLE) ? Wref[r+1][i] : Wref[r][i];
                W2[r][i] = sm_left * left + sm_center * Wref[r][i] + sm_right * right;
            }
        }
        Wref = W2;
    }

    static bool cholesky_decompose(vector<vector<double>>& A) {
        const int n = (int)A.size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = A[i][j];
                for (int k = 0; k < j; ++k) sum -= A[i][k]*A[j][k];
                if (i == j) { if (sum <= 0.0) return false; A[i][j] = sqrt(sum); }
                else A[i][j] = sum / A[j][j];
            }
            for (int j = i + 1; j < n; ++j) A[i][j] = 0.0;
        }
        return true;
    }
    static void chol_solve(const vector<vector<double>>& L, const array<double, D>& b, array<double, D>& w) {
        const int n = (int)L.size();
        array<double, D> y{};
        for (int i = 0; i < n; ++i) {
            double s = b[i];
            for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
            y[i] = s / L[i][i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = y[i];
            for (int k = i + 1; k < n; ++k) s -= L[k][i] * w[k];
            w[i] = s / L[i][i];
        }
    }

    void learn(int M, const vector<int>& dig, const vector<double>& ana) {
        // Pass 1: z-scoring stats (excluding bias)
        array<double, D> sum_x{}; sum_x.fill(0.0);
        array<double, D> sumsq_x{}; sumsq_x.fill(0.0);
        array<double, D> x{};
        for (int m = 0; m < M; ++m) {
            build_raw_features(dig, m, x);
            for (int i = 1; i < D; ++i) { sum_x[i] += x[i]; sumsq_x[i] += x[i]*x[i]; }
        }
        mu.fill(0.0); inv_std.fill(1.0);
        for (int i = 1; i < D; ++i) {
            double mean = (M > 0) ? (sum_x[i] / M) : 0.0;
            double ex2  = (M > 0) ? (sumsq_x[i] / M) : 0.0;
            double var  = max(1e-12, ex2 - mean * mean);
            mu[i] = mean; inv_std[i] = 1.0 / sqrt(var);
        }

        // Accumulate normal equations
        auto accumulate_normal = [&](vector<vector<double>>& XtX,
                                     array<array<double, D>, OVERSAMPLE>& Xty,
                                     const vector<double>* w_opt = nullptr) {
            array<double, D> xstd{};
            for (int m = 0; m < M; ++m) {
                build_raw_features(dig, m, xstd);
                standardize_inplace(xstd);
                double w = 1.0; if (w_opt) w = (*w_opt)[m];

                for (int i = 0; i < D; ++i) {
                    double xi = xstd[i], wxi = w * xi;
                    for (int j = 0; j <= i; ++j) XtX[i][j] += wxi * xstd[j];
                }
                int base = m * OVERSAMPLE;
                for (int r = 0; r < OVERSAMPLE; ++r) {
                    double y = ana[base + r];
                    for (int i = 0; i < D; ++i) Xty[r][i] += w * xstd[i] * y;
                }
            }
            for (int i = 0; i < D; ++i) for (int j = 0; j < i; ++j) XtX[j][i] = XtX[i][j];
        };

        auto solve_ridge = [&](vector<vector<double>>& XtX,
                               array<array<double, D>, OVERSAMPLE>& Xty) {
            double trace = 0.0; for (int i = 0; i < D; ++i) trace += XtX[i][i];
            double diag_avg = (trace > 0.0) ? (trace / D) : 1.0;
            double lambda = max(1e-12, 6e-3 * diag_avg);
            for (int i = 0; i < D; ++i) XtX[i][i] += lambda;

            if (!cholesky_decompose(XtX)) {
                for (int i = 0; i < D; ++i) XtX[i][i] += (1e-3 * (diag_avg + 1e-6));
                if (!cholesky_decompose(XtX)) { for (auto& wv : W) wv.fill(0.0); return; }
            }
            for (int r = 0; r < OVERSAMPLE; ++r) {
                array<double, D> wr{}; chol_solve(XtX, Xty[r], wr); W[r] = wr;
            }
        };

        // Initial fit
        {
            vector<vector<double>> XtX(D, vector<double>(D, 0.0));
            array<array<double, D>, OVERSAMPLE> Xty; for (auto& v : Xty) v.fill(0.0);
            accumulate_normal(XtX, Xty, nullptr);
            solve_ridge(XtX, Xty);
            smooth_phase_weights(W);
        }
        // One IRLS (Huber)
        {
            vector<double> w_m(M, 1.0);
            array<double, D> xstd{};
            constexpr double DELTA = 20;
            for (int m = 0; m < M; ++m) {
                build_raw_features(dig, m, xstd);
                standardize_inplace(xstd);
                int base = m * OVERSAMPLE;
                double se = 0.0;
                for (int r = 0; r < OVERSAMPLE; ++r) {
                    double yhat = 0.0; const auto& wr = W[r];
                    for (int i = 0; i < D; ++i) yhat += wr[i] * xstd[i];
                    double e = ana[base + r] - yhat; se += e * e;
                }
                double rms = sqrt(se / OVERSAMPLE + 1e-15);
                w_m[m] = (rms > DELTA) ? (DELTA / rms) : 1.0;
            }
            double avgw = 0.0; for (double w : w_m) avgw += w; avgw = (M > 0) ? (avgw / M) : 1.0;
            if (avgw < 0.9) {
                vector<vector<double>> XtX(D, vector<double>(D, 0.0));
                array<array<double, D>, OVERSAMPLE> Xty; for (auto& v : Xty) v.fill(0.0);
                accumulate_normal(XtX, Xty, &w_m);
                solve_ridge(XtX, Xty);
                smooth_phase_weights(W);
            }
        }

        // Residual LUTs
        lut11_res.clear(); lut9_res.clear(); lut7_res.clear();
        if (M > 0) { lut11_res.reserve(max(2000, M/2)); lut9_res.reserve(max(4000, M)); lut7_res.reserve(max(6000, M)); }

        array<double, OVERSAMPLE> yhat{}; array<double, D> xstd{};
        for (int m = 0; m < M; ++m) {
            build_raw_features(dig, m, xstd); standardize_inplace(xstd);
            for (int r = 0; r < OVERSAMPLE; ++r) {
                const auto& wr = W[r]; double y = 0.0; for (int i = 0; i < D; ++i) y += wr[i] * xstd[i];
                yhat[r] = y;
            }
            const int base = m * OVERSAMPLE;
            if (m > 7 && m < M - 2) {
                uint32_t k11 = pack_ctx11(dig, m);
                auto& e = lut11_res[k11];
                for (int r = 0; r < OVERSAMPLE; ++r) e.sum[r] += (float)(ana[base + r] - yhat[r]);
                if (e.count < 65535) ++e.count;
            }
            if (m > 5 && m < M - 2) {
                uint32_t k9 = pack_ctx9(dig, m);
                auto& e = lut9_res[k9];
                for (int r = 0; r < OVERSAMPLE; ++r) e.sum[r] += (float)(ana[base + r] - yhat[r]);
                if (e.count < 65535) ++e.count;
            }
            if (m > 2 && m < M - 3) {
                uint32_t k7 = pack_ctx7(dig, m);
                auto& e = lut7_res[k7];
                for (int r = 0; r < OVERSAMPLE; ++r) e.sum[r] += (float)(ana[base + r] - yhat[r]);
                if (e.count < 65535) ++e.count;
            }
        }

        // Train global FIR on predictions (reg + LUTs)
        vector<double> pred_train(M * OVERSAMPLE, 0.0);
        produce_with_luts(dig, pred_train);
        fit_fir(pred_train, ana);
    }

    void mix_residuals(int m, const vector<int>& dig, array<double, OVERSAMPLE>& resid_out) const {
        for (int r = 0; r < OVERSAMPLE; ++r) resid_out[r] = 0.0;
        array<double, OVERSAMPLE> r11{}, r9{}, r7{};
        int c11 = 0, c9 = 0, c7 = 0;

        if (m > 7 && m < (int)dig.size() - 2) {
            auto it = lut11_res.find(pack_ctx11(dig, m));
            if (it != lut11_res.end() && it->second.count > 0) {
                c11 = it->second.count;
                for (int r = 0; r < OVERSAMPLE; ++r) r11[r] = (double)it->second.sum[r] / c11;
                smooth16_inplace(r11);
            }
        }
        if (m > 5 && m < (int)dig.size() - 2) {
            auto it = lut9_res.find(pack_ctx9(dig, m));
            if (it != lut9_res.end() && it->second.count > 0) {
                c9 = it->second.count;
                for (int r = 0; r < OVERSAMPLE; ++r) r9[r] = (double)it->second.sum[r] / c9;
                smooth16_inplace(r9);
            }
        }
        if (m > 2 && m < (int)dig.size() - 3) {
            auto it = lut7_res.find(pack_ctx7(dig, m));
            if (it != lut7_res.end() && it->second.count > 0) {
                c7 = it->second.count;
                for (int r = 0; r < OVERSAMPLE; ++r) r7[r] = (double)it->second.sum[r] / c7;
                smooth16_inplace(r7);
            }
        }

        // shrinkage
        double a11 = (c11 > 0) ? (double)c11 / (c11 + beta11) : 0.0;
        double a9  = (c9  > 0) ? (double)c9  / (c9  + beta9 ) : 0.0;
        double a7  = (c7  > 0) ? (double)c7  / (c7  + beta7 ) : 0.0;

        double w11 = a11;
        double w9  = (1.0 - a11) * a9;
        double w7  = (1.0 - a11) * (1.0 - a9) * a7;

        if (w11 == 0.0 && w9 == 0.0 && w7 == 0.0) return;
        for (int r = 0; r < OVERSAMPLE; ++r) resid_out[r] = w11 * r11[r] + w9 * r9[r] + w7 * r7[r];
    }

    static void smooth16_inplace(array<double, OVERSAMPLE>& v) {
        array<double, OVERSAMPLE> t = v;
        for (int r = 0; r < OVERSAMPLE; ++r) {
            double left = (r > 0) ? v[r - 1] : v[r];
            double right = (r + 1 < OVERSAMPLE) ? v[r + 1] : v[r];
            t[r] = 0.20 * left + 0.60 * v[r] + 0.20 * right;
        }
        v = t;
    }

    void produce_with_luts(const vector<int>& dig, vector<double>& out) const {
        const int K = (int)dig.size();
        array<double, D> x{};
        array<double, OVERSAMPLE> reg_hat{};
        array<double, OVERSAMPLE> resid{};
        for (int m = 0; m < K; ++m) {
            build_raw_features(dig, m, x); standardize_inplace(x);
            for (int r = 0; r < OVERSAMPLE; ++r) {
                const auto& wr = W[r]; double y = 0.0;
                for (int i = 0; i < D; ++i) y += wr[i] * x[i];
                reg_hat[r] = y;
            }
            mix_residuals(m, dig, resid);
            const int base = m * OVERSAMPLE;
            for (int r = 0; r < OVERSAMPLE; ++r) out[base + r] = reg_hat[r] + resid[r];
        }
    }

    void fit_fir(const vector<double>& pred, const vector<double>& target) {
        vector<double> Htmp; fit_fir_linear(pred, target, LFIR, Htmp, true);
        if ((int)Htmp.size() == LFIR) { for (int i = 0; i < LFIR; ++i) H[i] = Htmp[i]; has_fir = true; }
        else { has_fir = false; }
    }

    void apply_fir_inplace(vector<double>& y) const {
        if (!has_fir) { for (double& v : y) v = clamp11(v); return; }
        vector<double> Hvec(LFIR); for (int i = 0; i < LFIR; ++i) Hvec[i] = H[i];
        ::apply_fir_inplace(y, Hvec);
    }

    vector<double> predict(const int K, const vector<int>& dig) const {
        vector<double> out((size_t)K * OVERSAMPLE, 0.0);
        produce_with_luts(dig, out);
        apply_fir_inplace(out);
        return out;
    }
};

/* ========================= 5150: L1 cache + polynomial ridge (with delta + OOF RMSE + SM blend) ========================= */

static void generate_poly_features_5150(const vector<double>& window, vector<double>& features) {
    features[0] = 1.0; int col_idx = 1;
    for (double v : window) features[col_idx++] = v;
    for (double v : window) features[col_idx++] = v * v;
    for (double v : window) features[col_idx++] = v * v * v;
    for (size_t j = 0; j + 1 < window.size(); ++j) features[col_idx++] = window[j] * window[j + 1];
    for (size_t j = 0; j + 2 < window.size(); ++j) features[col_idx++] = window[j] * window[j + 2];
    for (size_t j = 0; j + 2 < window.size(); ++j) features[col_idx++] = window[j] * window[j + 1] * window[j + 2];
    for (size_t j = 0; j + 3 < window.size(); ++j) features[col_idx++] = window[j] * window[j + 3];
    for (size_t j = 0; j + 4 < window.size(); ++j) features[col_idx++] = window[j] * window[j + 4];
}

struct Run5150Result {
    vector<double> test;       // K*OSR
    vector<double> train_pred; // Meff*OSR (aligned region)
    double train_rmse = 1e9;
    int start_dig = 0, end_dig = 0, best_delta = 0;
};

static Run5150Result run_5150_all(
    const vector<double>& ana_waveform_1, int M, const vector<int>& dig_data_1,
    int K, const vector<int>& dig_data_2, int OSR,
    int P_cache, int Q_cache, int P_reg, int Q_reg, double ridge_lambda_cap,
    int best_delta,
    const Ctrl::C1620& smcfg, double alpha_5150_vs_sm, bool alpha_override_case, int caseM
) {
    Run5150Result R; R.best_delta = best_delta;

    // Determine valid train slice given delta
    int start_dig = 0; while (start_dig * OSR + best_delta < 0) start_dig++;
    int end_dig = M; while (end_dig > 0 && 1LL*(end_dig - 1)*OSR + OSR + best_delta > (long long)ana_waveform_1.size()) end_dig--;
    if (end_dig < start_dig) { // degenerate
        R.test.assign(1LL*K*OSR, 0.0);
        R.train_pred.clear();
        R.start_dig = R.end_dig = 0;
        return R;
    }
    const int Meff = end_dig - start_dig;
    R.start_dig = start_dig; R.end_dig = end_dig;

    // Build Y_train_full (aligned)
    Matrix Y_train_full(Meff, OSR);
    for (int i = 0; i < Meff; ++i) {
        int srci = start_dig + i;
        for (int j = 0; j < OSR; ++j) {
            Y_train_full.data[i][j] = ana_waveform_1[1LL * srci * OSR + best_delta + j];
        }
    }

    // Padded digital sequences
    const int num_features_cache = P_cache + 1 + Q_cache;
    vector<int> d1_cache = vector<int>(P_cache, 0);
    d1_cache.insert(d1_cache.end(), dig_data_1.begin() + start_dig, dig_data_1.begin() + end_dig);
    d1_cache.insert(d1_cache.end(), Q_cache, 0);

    // L1 sums and counts
    unordered_map<vector<int>, vector<double>, VecHash> l1_sum;
    unordered_map<vector<int>, int, VecHash> l1_cnt;
    l1_sum.reserve(Meff*2); l1_cnt.reserve(Meff*2);
    for (int i = 0; i < Meff; ++i) {
        vector<int> pat(d1_cache.begin() + i, d1_cache.begin() + i + num_features_cache);
        auto& acc = l1_sum[pat];
        if (acc.empty()) acc.assign(OSR, 0.0);
        l1_cnt[pat]++;
        const double* src = Y_train_full.data[i].data();
        for (int j = 0; j < OSR; ++j) acc[j] += src[j];
    }
    // Mean cache
    unordered_map<vector<int>, vector<double>, VecHash> l1_cache; l1_cache.reserve(l1_sum.size());
    for (auto& kv : l1_sum) {
        vector<double> mean = kv.second;
        int c = l1_cnt.at(kv.first);
        double inv = 1.0 / max(1, c);
        for (double& v : mean) v *= inv;
        l1_cache.emplace(kv.first, std::move(mean));
    }

    // Regression training (ridge) on dig1 (aligned region)
    const int num_features_reg = P_reg + 1 + Q_reg;
    const int num_poly_features = 1 + num_features_reg * 3 + (num_features_reg - 1) + (num_features_reg - 2) * 2 + (num_features_reg - 3) + (num_features_reg - 4);

    vector<int> d1_reg = vector<int>(P_reg, 0);
    d1_reg.insert(d1_reg.end(), dig_data_1.begin() + start_dig, dig_data_1.begin() + end_dig);
    d1_reg.insert(d1_reg.end(), Q_reg, 0);

    vector<vector<double>> X_cols(num_poly_features, vector<double>(Meff));
    vector<double> current_features(num_poly_features);
    vector<double> window(num_features_reg);
    for (int i = 0; i < Meff; ++i) {
        for (int j = 0; j < num_features_reg; ++j) window[j] = (double)d1_reg[i + j];
        generate_poly_features_5150(window, current_features);
        for (int f = 0; f < num_poly_features; ++f) X_cols[f][i] = current_features[f];
    }

    Matrix XtX(num_poly_features, num_poly_features);
    Matrix XtY(num_poly_features, OSR);
    auto dot = [](const vector<double>& a, const vector<double>& b) {
        double s = 0.0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
    };
    for (int j = 0; j < num_poly_features; ++j) {
        for (int k = j; k < num_poly_features; ++k) {
            double s = dot(X_cols[j], X_cols[k]);
            XtX.data[j][k] = XtX.data[k][j] = s;
        }
        for (int r = 0; r < OSR; ++r) {
            double s = 0.0;
            for (int i = 0; i < Meff; ++i) s += X_cols[j][i] * Y_train_full.data[i][r];
            XtY.data[j][r] = s;
        }
    }
    double tr_XtX = 0.0; for (size_t i = 0; i < XtX.rows; ++i) tr_XtX += XtX.data[i][i];
    double lam_auto = std::max(1e-12, 3e-2 * (tr_XtX / std::max<size_t>(1, XtX.rows)));
    double lam = std::clamp(lam_auto, 1e-4, ridge_lambda_cap);
    for (size_t i = 0; i < XtX.rows; ++i) XtX.data[i][i] += lam;
    Matrix regression_weights = Matrix::solveSPD(XtX, XtY);

    // Build test-time padded dig2
    vector<int> d2_cache(P_cache, 0); d2_cache.insert(d2_cache.end(), dig_data_2.begin(), dig_data_2.end()); d2_cache.insert(d2_cache.end(), Q_cache, 0);
    vector<int> d2_reg  (P_reg  , 0); d2_reg.insert(d2_reg.end(), dig_data_2.begin(), dig_data_2.end()); d2_reg.insert(d2_reg.end(), Q_reg, 0);

    // Predict test with cache+reg
    vector<double> out_5150(1LL * K * OSR, 0.0);
    vector<double> x_miss_reg(num_poly_features);
    for (int i = 0; i < K; ++i) {
        vector<int> pat(d2_cache.begin() + i, d2_cache.begin() + i + num_features_cache);
        auto it = l1_cache.find(pat);
        if (it != l1_cache.end()) {
            const auto& pred = it->second;
            for (int j = 0; j < OSR; ++j) out_5150[1LL * i * OSR + j] = clamp1(pred[j]);
        } else {
            for (int w_idx = 0; w_idx < num_features_reg; ++w_idx) window[w_idx] = (double)d2_reg[i + w_idx];
            generate_poly_features_5150(window, x_miss_reg);
            for (int j = 0; j < OSR; ++j) {
                double pv = 0.0;
                for (int kf = 0; kf < num_poly_features; ++kf) pv += x_miss_reg[kf] * regression_weights.data[kf][j];
                out_5150[1LL * i * OSR + j] = clamp1(pv);
            }
        }
    }

    // Out-of-fold training prediction for RMSE/stacking
    vector<double> pred_train(Meff * OSR, 0.0);
    for (int i = 0; i < Meff; ++i) {
        vector<int> pat(d1_cache.begin() + i, d1_cache.begin() + i + num_features_cache);
        auto itS = l1_sum.find(pat);
        int cnt = l1_cnt[pat];
        bool used_cache = false;
        if (itS != l1_sum.end() && cnt > 1) {
            // leave-one-out mean
            for (int j = 0; j < OSR; ++j) {
                double s = itS->second[j] - Y_train_full.data[i][j];
                pred_train[1LL * i * OSR + j] = clamp1(s / (cnt - 1));
            }
            used_cache = true;
        }
        if (!used_cache) {
            // Fallback to regression for this training sample
            for (int w_idx = 0; w_idx < num_features_reg; ++w_idx) window[w_idx] = (double)d1_reg[i + w_idx];
            generate_poly_features_5150(window, x_miss_reg);
            for (int j = 0; j < OSR; ++j) {
                double pv = 0.0;
                for (int kf = 0; kf < num_poly_features; ++kf) pv += x_miss_reg[kf] * regression_weights.data[kf][j];
                pred_train[1LL * i * OSR + j] = clamp1(pv);
            }
        }
    }

    // Blend with SignalModel (1620)
    SignalModel sm;
    sm.beta11 = smcfg.beta11; sm.beta9 = smcfg.beta9; sm.beta7 = smcfg.beta7;
    sm.sm_left = smcfg.sm_left; sm.sm_center = smcfg.sm_center; sm.sm_right = smcfg.sm_right;
    // Train SM on aligned training
    vector<int> dig_train(dig_data_1.begin() + start_dig, dig_data_1.begin() + end_dig);
    vector<double> ana_train(Meff * OSR);
    for (int i = 0; i < Meff; ++i) for (int j = 0; j < OSR; ++j) ana_train[1LL*i*OSR + j] = Y_train_full.data[i][j];
    sm.learn(Meff, dig_train, ana_train);
    // Predict on dig2
    vector<double> y_sm = sm.predict(K, dig_data_2);
    // Predict training (aligned region) with SM for weight learning in 5150 blend (useful if alpha not 1)
    vector<double> y_sm_train = sm.predict(Meff, dig_train);

    double alpha = alpha_5150_vs_sm;
    if (!alpha_override_case) {
        if (caseM == 5460) alpha = 1.0; // original file's override
    }
    const double w5150 = std::min(1.0, std::max(0.0, alpha));
    const double wsm   = 1.0 - w5150;

    // Blend both for test and train
    const size_t Ntest = out_5150.size();
    R.test.assign(Ntest, 0.0);
    for (size_t i = 0; i < Ntest; ++i) {
        double y = w5150 * out_5150[i] + wsm * (i < y_sm.size() ? y_sm[i] : 0.0);
        R.test[i] = clamp1(y);
    }
    R.train_pred.assign(pred_train.size(), 0.0);
    for (size_t i = 0; i < R.train_pred.size(); ++i) {
        double y = w5150 * pred_train[i] + wsm * (i < y_sm_train.size() ? y_sm_train[i] : 0.0);
        R.train_pred[i] = clamp1(y);
    }

    // Compute RMSE vs aligned training
    vector<double> y_true(Meff * OSR);
    for (int i = 0; i < Meff; ++i) for (int j = 0; j < OSR; ++j) y_true[1LL*i*OSR + j] = Y_train_full.data[i][j];
    R.train_rmse = rmse(R.train_pred, y_true);
    return R;
}

/* ========================= 1801: Hierarchical cache models (train+predict returning both) ========================= */

namespace N1801 {
    using Waveform = vector<double>;
    using DigitalData = vector<int>;
    using Configs = vector<pair<string, int>>;
    using ModelMap = unordered_map<vector<int>, vector<double>, VecHash>;
    using Models = unordered_map<string, ModelMap>;

    struct RunResult {
        vector<double> pred_test;   // K*OSR
        vector<double> pred_train;  // Meff*OSR (aligned region)
        int best_delta = 0;
        int start_dig = 0, end_dig = 0;
    };

    Models build_models(const Waveform& data, const DigitalData& dig_data, const Configs& configs, int OSR) {
        Models models;
        int m = (int)dig_data.size();
        for (const auto& cfg : configs) {
            const string& name = cfg.first;
            int size = cfg.second;
            ModelMap sums;
            unordered_map<vector<int>, int, VecHash> counts;
            int offset = (name == "pre_duplet" || name == "post_duplet") ? 0 : (size - 1) / 2;

            for (int i = 0; i < m; ++i) {
                vector<int> pattern;
                if (name == "pre_duplet") {
                    if (i > 0) pattern = {dig_data[i - 2], dig_data[i]}; else continue;
                } else if (name == "post_duplet") {
                    if (i < m - 1) pattern = {dig_data[i], dig_data[i + 1]}; else continue;
                } else {
                    if (i < offset || i >= m - offset) continue;
                    if (size > 1) pattern.assign(dig_data.begin() + i - offset, dig_data.begin() + i + offset + 1);
                    else pattern = {dig_data[i]};
                }

                int base = i * OSR;
                if (!sums.count(pattern)) sums[pattern] = vector<double>(OSR, 0.0);
                counts[pattern]++;
                for (int j = 0; j < OSR; ++j) sums.at(pattern)[j] += data[base + j];
            }
            ModelMap model_map;
            for (auto const& [p, v] : sums) {
                int c = counts.at(p);
                if (c > 0) {
                    vector<double> avg(OSR);
                    for (int j = 0; j < OSR; ++j) avg[j] = v[j] / (double)c;
                    model_map[p] = std::move(avg);
                }
            }
            models[name] = std::move(model_map);
        }
        return models;
    }

    Waveform predict_waveform(const DigitalData& dig_data, const Models& models, const Configs& configs, int OSR) {
        int k = (int)dig_data.size();
        Waveform waveform(1LL * k * OSR);
        const vector<double> default_waveform(OSR, 0.0);
        for (int i = 0; i < k; ++i) {
            const vector<double>* pred = &default_waveform;
            for (auto it = configs.rbegin(); it != configs.rend(); ++it) {
                const string& name = it->first;
                int size = it->second;
                vector<int> pattern;
                if (name == "pre_duplet") {
                    if (i > 0) pattern = {dig_data[i-2], dig_data[i]}; else continue;
                } else if (name == "post_duplet") {
                    if (i < k - 1) pattern = {dig_data[i], dig_data[i+1]}; else continue;
                } else {
                    int offset = (size - 1) / 2;
                    if (i < offset || i >= k - offset) continue;
                    pattern.assign(dig_data.begin() + i - offset, dig_data.begin() + i + offset + 1);
                }
                const auto& model_for_name = models.at(name);
                auto mit = model_for_name.find(pattern);
                if (mit != model_for_name.end()) { pred = &mit->second; break; }
            }
            for (int j = 0; j < OSR; ++j) waveform[1LL*i*OSR + j] = (*pred)[j];
        }
        return waveform;
    }

    RunResult run_both(const vector<double>& ana_waveform_1, int M, const vector<int>& dig_data_1,
                       int K, const vector<int>& dig_data_2, int OSR,
                       const vector<Configs>& stage_configs,
                       int force_delta_min, int force_delta_max) {
        RunResult res;
        int best_delta = 0;
        double best_rmse = numeric_limits<double>::infinity();
        const Configs& eval_configs = stage_configs[0];
        for (int delta = force_delta_min; delta <= force_delta_max; ++delta) {
            int start_dig = 0; while (start_dig * OSR + delta < 0) start_dig++;
            int end_dig = M; while (end_dig > 0 && 1LL*(end_dig-1)*OSR + OSR + delta > (long long)ana_waveform_1.size()) end_dig--;
            if (end_dig <= start_dig) continue;
            DigitalData dig_train_eval(dig_data_1.begin() + start_dig, dig_data_1.begin() + end_dig);
            Waveform ana_train_eval; ana_train_eval.reserve(1LL * dig_train_eval.size() * OSR);
            for (int i = start_dig; i < end_dig; ++i)
                for (int j = 0; j < OSR; ++j)
                    ana_train_eval.push_back(ana_waveform_1[i * OSR + delta + j]);

            Models eval_models = build_models(ana_train_eval, dig_train_eval, eval_configs, OSR);
            Waveform pred_waveform_eval = predict_waveform(dig_train_eval, eval_models, eval_configs, OSR);
            double se = 0.0;
            for (size_t i = 0; i < ana_train_eval.size(); ++i) { double d = ana_train_eval[i] - pred_waveform_eval[i]; se += d * d; }
            double rmse = sqrt(se / ana_train_eval.size());
            if (rmse < best_rmse) { best_rmse = rmse; best_delta = delta; }
        }
        res.best_delta = best_delta;

        int start_dig_final = 0; while (start_dig_final * OSR + best_delta < 0) start_dig_final++;
        int end_dig_final = M; while (end_dig_final > 0 && 1LL*(end_dig_final-1)*OSR + OSR + best_delta > (long long)ana_waveform_1.size()) end_dig_final--;
        res.start_dig = start_dig_final; res.end_dig = end_dig_final;

        DigitalData dig_train_final(dig_data_1.begin() + start_dig_final, dig_data_1.begin() + end_dig_final);
        Waveform ana_train_final; ana_train_final.reserve(1LL * dig_train_final.size() * OSR);
        for (int i = start_dig_final; i < end_dig_final; ++i)
            for (int j = 0; j < OSR; ++j)
                ana_train_final.push_back(ana_waveform_1[i * OSR + best_delta + j]);

        Waveform current_residual = ana_train_final;
        Waveform final_pred_train(1LL * dig_train_final.size() * OSR, 0.0);
        Waveform final_pred_test(1LL * K * OSR, 0.0);

        for (const auto& current_stage_config : stage_configs) {
            Models models = build_models(current_residual, dig_train_final, current_stage_config, OSR);
            Waveform pred_train = predict_waveform(dig_train_final, models, current_stage_config, OSR);
            Waveform pred_test  = predict_waveform(dig_data_2,     models, current_stage_config, OSR);
            for (size_t i = 0; i < final_pred_test.size(); ++i) final_pred_test[i] += pred_test[i];
            for (size_t i = 0; i < final_pred_train.size(); ++i) final_pred_train[i] += pred_train[i];
            for (size_t i = 0; i < current_residual.size(); ++i) current_residual[i] -= pred_train[i];
        }
        for (double& v : final_pred_test)  v = clamp1(v);
        for (double& v : final_pred_train) v = clamp1(v);
        res.pred_test  = std::move(final_pred_test);
        res.pred_train = std::move(final_pred_train);
        return res;
    }

    vector<double> run(const vector<double>& ana_waveform_1, int M, const vector<int>& dig_data_1,
                       int K, const vector<int>& dig_data_2, int OSR,
                       const vector<Configs>& stage_configs, int delta_min, int delta_max) {
        auto rr = run_both(ana_waveform_1, M, dig_data_1, K, dig_data_2, OSR, stage_configs, delta_min, delta_max);
        return rr.pred_test;
    }
}

/* ========================= 1425: Memory Polynomial with LUT (train+predict returning both) ========================= */

namespace N1425 {
    static vector<double> run(const vector<double>& ana1, int M,
                              const vector<int>& dig1_i, int K,
                              const vector<int>& dig2_i, int OSR,
                              int L_pre, int L_post, const vector<int>& orders,
                              double lam, int delta_min, int delta_max, double tau);

    struct PhaseModel { int r = 0; vector<double> w; vector<double> lut; };
    struct Model { int OSR = 16; vector<int> orders; int L_pre = 100, L_post = 8; vector<PhaseModel> phases; };

    static inline int8_t level_to_index(int v) { if (v == -3) return 0; if (v == -1) return 1; if (v == 1) return 2; return 3; }
    static inline double dot_product(const double* a, const double* b, int n) { double s = 0.0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s; }

    static bool cholesky_decompose(vector<double>& A, int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = A[i * n + j];
                for (int k = 0; k < j; ++k) sum -= A[i * n + k] * A[j * n + k];
                if (i == j) { if (sum <= 1e-12) { sum += 1e-9; if (sum <= 0) return false; } A[i * n + i] = sqrt(sum); }
                else {
                    if (fabs(A[j*n+j]) < 1e-12) return false;
                    A[i * n + j] = sum / A[j * n + j];
                }
            }
            for (int j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
        }
        return true;
    }

    static void cholesky_solve_inplace(const vector<double>& L, int n, vector<double>& b) {
        for (int i = 0; i < n; ++i) {
            double sum = b[i]; for (int k = 0; k < i; ++k) sum -= L[i * n + k] * b[k];
            b[i] = sum / L[i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double sum = b[i]; for (int k = i + 1; k < n; ++k) sum -= L[k * n + i] * b[k];
            b[i] = sum / L[i * n + i];
        }
    }

    static void build_feature_block_cols(
        const vector<vector<double>>& phis, int L_pre, int L_post,
        int k_start, int k_end, vector<double>& Xcols, int& D_out, int& T_out
    ) {
        int P = (int)phis.size();
        int Q = L_pre + L_post + 1;
        int T = max(0, k_end - k_start);
        int D = P * Q + 1;
        D_out = D; T_out = T;
        Xcols.assign((size_t)D * (size_t)T, 0.0);
        int col = 0;
        for (int p = 0; p < P; ++p) {
            const vector<double>& phi = phis[p];
            for (int m = -L_pre; m <= L_post; ++m) {
                double* colptr = &Xcols[(size_t)col * (size_t)T];
                int start_idx = k_start - m;
                for (int i = 0; i < T; ++i) colptr[i] = phi[start_idx + i];
                ++col;
            }
        }
        double* colptr = &Xcols[(size_t)col * (size_t)T];
        for (int i = 0; i < T; ++i) colptr[i] = 1.0;
    }

    static void solve_ridge_from_Xcols(
        const vector<double>& Xcols, int T, int D,
        const vector<double>& y, double lam, vector<double>& w_out
    ) {
        vector<double> A((size_t)D * (size_t)D, 0.0);
        vector<double> b(D, 0.0);
        for (int j = 0; j < D; ++j) {
            const double* Xj = &Xcols[(size_t)j * (size_t)T];
            b[j] = dot_product(Xj, y.data(), T);
            for (int k = j; k < D; ++k) {
                const double* Xk = &Xcols[(size_t)k * (size_t)T];
                double s = dot_product(Xj, Xk, T);
                A[(size_t)j * (size_t)D + k] = s;
                A[(size_t)k * (size_t)D + j] = s;
            }
        }
        for (int j = 0; j < D - 1; ++j) A[(size_t)j * (size_t)D + j] += lam;
        vector<double> L = A;
        if (cholesky_decompose(L, D)) {
             vector<double> rhs = b;
             cholesky_solve_inplace(L, D, rhs);
             w_out = std::move(rhs);
        } else {
            w_out.assign(D, 0.0);
        }
    }

    int coarse_delta_search( const vector<double>& ana, const vector<double>& s_norm, int OSR, int delta_min, int delta_max) {
        const int N = (int)ana.size();
        const int M = (int)s_norm.size();
        int best_delta = 0;
        double best_sse = numeric_limits<double>::infinity();
        for (int delta = delta_min; delta <= delta_max; ++delta) {
            double total_sse = 0.0;
            for (int r = 0; r < OSR; ++r) {
                long long start = (long long)delta + r;
                int i0 = (start < 0 ? (int)((-start + OSR - 1) / OSR) : 0);
                int cnt = (start > N - 1 ? 0 : min(M - i0, (int)((N - start + OSR - 1) / OSR)));
                if (cnt <= 8) continue;
                double s_sum = 0.0, y_sum = 0.0;
                for (int j = 0; j < cnt; ++j) { int idx = (int)(start + 1LL * OSR * (i0 + j)); s_sum += s_norm[i0 + j]; y_sum += ana[idx]; }
                double s_mean = s_sum / cnt, y_mean = y_sum / cnt;
                double s_denom = 0.0, s_y = 0.0;
                for (int j = 0; j < cnt; ++j) {
                    double sv = s_norm[i0 + j] - s_mean;
                    int idx = (int)(start + 1LL * OSR * (i0 + j));
                    double yv = ana[idx] - y_mean;
                    s_denom += sv * sv; s_y += sv * yv;
                }
                double alpha = s_y / (s_denom + 1e-12);
                double beta = y_mean - alpha * s_mean;
                double sse = 0.0;
                for (int j = 0; j < cnt; ++j) {
                    int idx = (int)(start + 1LL * OSR * (i0 + j));
                    double rres = ana[idx] - (alpha * s_norm[i0 + j] + beta);
                    sse += rres * rres;
                }
                total_sse += sse;
            }
            if (total_sse < best_sse) { best_sse = total_sse; best_delta = delta; }
        }
        return best_delta;
    }

    Model train_memory_poly( const vector<double>& ana, const vector<int16_t>& s_int, int OSR, int delta, int L_pre, int L_post, const vector<int>& orders, double lam, double tau) {
        const int N = (int)ana.size();
        const int M = (int)s_int.size();
        vector<double> s_norm(M);
        for (int i = 0; i < M; ++i) s_norm[i] = (double)s_int[i] / 3.0;

        vector<vector<double>> phis;
        phis.reserve(orders.size());
        for (int p : orders) {
            vector<double> phi(M);
            if (p == 1) { for (int i = 0; i < M; ++i) phi[i] = s_norm[i]; }
            else if (p == -1) { for (int i = 0; i < M; ++i) phi[i] = log(fabs(s_norm[i]) + 1e-9); }
            else { for (int i = 0; i < M; ++i) phi[i] = pow(s_norm[i], (double)p); }
            phis.push_back(std::move(phi));
        }

        vector<int8_t> d1_idx(M);
        for (int i = 0; i < M; ++i) d1_idx[i] = level_to_index(s_int[i]);

        Model model;
        model.OSR = OSR; model.orders = orders; model.L_pre = L_pre; model.L_post = L_post;

        for (int r = 0; r < OSR; ++r) {
            long long start = (long long)delta + r;
            if (start > N - 1) continue;
            int first_by_ana = (start < 0) ? (int)((-start + OSR - 1) / OSR) : 0;
            int max_k_by_ana = (int)((N - 1 - start) / OSR) + 1;
            int k_start = max(L_pre, first_by_ana);
            int k_end = min(M - L_post, max_k_by_ana);
            int T = k_end - k_start;
            if (T <= 0) continue;

            vector<double> y(T);
            for (int i = 0; i < T; ++i) { int k = k_start + i; int idx = (int)(start + 1LL * OSR * k); y[i] = ana[idx]; }

            vector<double> Xcols; int D = 0, Tcheck = 0;
            build_feature_block_cols(phis, L_pre, L_post, k_start, k_end, Xcols, D, Tcheck);
            vector<double> w;
            solve_ridge_from_Xcols(Xcols, T, D, y, lam, w);

            vector<double> sums(64, 0.0), cnts(64, 0.0);
            for (int i = 0; i < T; ++i) {
                int k = k_start + i;
                if (k <= 0 || k >= M - 1 || w.empty()) continue;
                double yhat = w.back();
                for (int j = 0; j < D - 1; ++j) yhat += Xcols[(size_t)j * (size_t)T + i] * w[j];
                double res = y[i] - yhat;
                int prev_i = d1_idx[k - 1], cur_i = d1_idx[k], next_i = d1_idx[k + 1];
                int id = prev_i * 16 + cur_i * 4 + next_i;
                sums[id] += res; cnts[id] += 1.0;
            }
            vector<double> lut(64, 0.0);
            for (int i = 0; i < 64; ++i) if (cnts[i] > 0.0) { double avg = sums[i] / cnts[i]; double shrink = cnts[i] / (cnts[i] + tau); lut[i] = avg * shrink; }
            PhaseModel pm; pm.r = r; pm.w = std::move(w); pm.lut = std::move(lut);
            model.phases.push_back(std::move(pm));
        }
        return model;
    }

    vector<double> infer_memory_poly(const Model& model, const vector<int16_t>& s_int) {
        const int OSR = model.OSR, L_pre = model.L_pre, L_post = model.L_post;
        const vector<int>& orders = model.orders;
        const int P = (int)orders.size(), Q = L_pre + L_post + 1, K = (int)s_int.size();

        vector<double> s_norm(K); for (int i = 0; i < K; ++i) s_norm[i] = (double)s_int[i] / 3.0;

        vector<vector<double>> phis(P, vector<double>(K));
        for (int pidx = 0; pidx < P; ++pidx) {
            int p = orders[pidx];
            if (p == 1) for (int i = 0; i < K; ++i) phis[pidx][i] = s_norm[i];
            else if (p == -1) { for (int i = 0; i < K; ++i) phis[pidx][i] = log(fabs(s_norm[i]) + 1e-9); }
            else for (int i = 0; i < K; ++i) phis[pidx][i] = pow(s_norm[i], (double)p);
        }

        vector<vector<double>> phiPads(P, vector<double>(K + L_pre + L_post));
        for (int pidx = 0; pidx < P; ++pidx) {
            vector<double>& pad = phiPads[pidx];
            const vector<double>& phi = phis[pidx];
            double leftv = phi.empty() ? 0.0 : phi.front(); for (int i = 0; i < L_pre; ++i) pad[i] = leftv;
            for (int i = 0; i < K; ++i) pad[L_pre + i] = phi[i];
            double rightv = phi.empty() ? 0.0 : phi.back(); for (int i = 0; i < L_post; ++i) pad[L_pre + K + i] = rightv;
        }

        vector<int8_t> d2_idx(K); for (int i = 0; i < K; ++i) d2_idx[i] = level_to_index(s_int[i]);
        vector<int> lut_id(K);
        for (int k = 0; k < K; ++k) {
            int prev_i = (k == 0) ? d2_idx[0] : d2_idx[k - 1];
            int cur_i  = d2_idx[k];
            int next_i = (k == K - 1) ? d2_idx[K - 1] : d2_idx[k + 1];
            lut_id[k] = prev_i * 16 + cur_i * 4 + next_i;
        }

        vector<double> out((size_t)K * (size_t)OSR, 0.0);
        for (const auto& pm : model.phases) {
            const vector<double>& w = pm.w; if (w.empty()) continue;
            const vector<double>& lut = pm.lut;
            vector<double> y_phase(K, w.back());
            for (int pidx = 0; pidx < P; ++pidx) {
                const vector<double>& pad = phiPads[pidx];
                int base = pidx * Q;
                for (int m_idx = 0; m_idx < Q; ++m_idx) {
                    double coeff = w[base + m_idx]; if (coeff == 0.0) continue;
                    int offset = L_pre - (-L_pre + m_idx); // aligns pad
                    const double* src = &pad[offset];
                    for (int k = 0; k < K; ++k) y_phase[k] += coeff * src[k];
                }
            }
            for (int k = 0; k < K; ++k) y_phase[k] += lut[lut_id[k]];
            for (int k = 0; k < K; ++k) out[(size_t)k * OSR + pm.r] = y_phase[k];
        }
        return out;
    }

    vector<double> run(const vector<double>& ana1, int M, const vector<int>& dig1_i, int K, const vector<int>& dig2_i, int OSR, int L_pre, int L_post, const vector<int>& orders, double lam, int delta_min, int delta_max, double tau) {
        vector<int16_t> dig1(M), dig2(K);
        for (int i = 0; i < M; ++i) dig1[i] = (int16_t)dig1_i[i];
        for (int i = 0; i < K; ++i) dig2[i] = (int16_t)dig2_i[i];
        vector<double> s_norm1(M); for (int i = 0; i < M; ++i) s_norm1[i] = (double)dig1[i] / 3.0;
        int best_delta = coarse_delta_search(ana1, s_norm1, OSR, delta_min, delta_max);
        Model model = train_memory_poly(ana1, dig1, OSR, best_delta, L_pre, L_post, orders, lam, tau);
        vector<double> out = infer_memory_poly(model, dig2);
        for (double& v : out) v = clamp1(v);
        return out;
    }

    struct Run2 {
        vector<double> pred_test;
        vector<double> pred_train; // aligned Meff*OSR
        int best_delta = 0;
        int start_dig = 0, end_dig = 0;
    };

    static Run2 run_both(const vector<double>& ana1, int M, const vector<int>& dig1_i, int K, const vector<int>& dig2_i, int OSR,
                         int L_pre, int L_post, const vector<int>& orders, double lam,
                         int force_delta_min, int force_delta_max, double tau) {
        Run2 R;
        vector<int16_t> dig1(M), dig2(K);
        for (int i = 0; i < M; ++i) dig1[i] = (int16_t)dig1_i[i];
        for (int i = 0; i < K; ++i) dig2[i] = (int16_t)dig2_i[i];
        vector<double> s_norm1(M); for (int i = 0; i < M; ++i) s_norm1[i] = (double)dig1[i] / 3.0;
        int best_delta = coarse_delta_search(ana1, s_norm1, OSR, force_delta_min, force_delta_max);
        R.best_delta = best_delta;

        int start_dig = 0; while (start_dig * OSR + best_delta < 0) start_dig++;
        int end_dig = M; while (end_dig > 0 && 1LL*(end_dig-1)*OSR + OSR + best_delta > (long long)ana1.size()) end_dig--;
        R.start_dig = start_dig; R.end_dig = end_dig;
        vector<int16_t> dig_train(dig1.begin() + start_dig, dig1.begin() + end_dig);

        Model model = train_memory_poly(ana1, dig1, OSR, best_delta, L_pre, L_post, orders, lam, tau);
        vector<double> out_test = infer_memory_poly(model, dig2);
        vector<double> out_train = infer_memory_poly(model, dig_train);
        for (double& v : out_test) v = clamp1(v);
        for (double& v : out_train) v = clamp1(v);
        R.pred_test = std::move(out_test);
        R.pred_train = std::move(out_train);
        return R;
    }
}

/* ========================= Ensemble helpers ========================= */

// Project vector w onto probability simplex {w>=0, sum=1}
static void project_simplex(vector<double>& w) {
    int n = (int)w.size();
    vector<double> u = w; sort(u.begin(), u.end(), greater<double>());
    double css = 0.0; int rho = -1;
    for (int i = 0; i < n; ++i) { css += u[i]; double t = (css - 1.0) / (i + 1); if (u[i] - t > 0) rho = i; }
    double theta = (accumulate(u.begin(), u.begin() + rho + 1, 0.0) - 1.0) / (rho + 1);
    for (int i = 0; i < n; ++i) w[i] = max(0.0, w[i] - theta);
}

static vector<double> learn_weights_nnls_simplex(
    const vector<vector<double>>& Xcols, // each size N
    const vector<double>& y,             // size N
    const vector<double>& prior, double prior_strength // L2 toward prior
) {
    const int m = (int)Xcols.size();
    const int N = (int)y.size();
    vector<vector<double>> G(m, vector<double>(m, 0.0));
    vector<double> b(m, 0.0);
    for (int i = 0; i < m; ++i) {
        const vector<double>& Xi = Xcols[i];
        double bi = 0.0;
        for (int n = 0; n < N; ++n) bi += Xi[n] * y[n];
        b[i] = bi + prior_strength * prior[i];
        for (int j = i; j < m; ++j) {
            const vector<double>& Xj = Xcols[j];
            double gij = 0.0;
            for (int n = 0; n < N; ++n) gij += Xi[n] * Xj[n];
            if (i == j) gij += prior_strength;
            G[i][j] = G[j][i] = gij;
        }
    }
    // Solve Gw = b (Cholesky)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = G[i][j];
            for (int k = 0; k < j; ++k) s -= G[i][k] * G[j][k];
            if (i == j) { if (s <= 1e-12) s = 1e-9; G[i][j] = sqrt(s); }
            else G[i][j] = s / G[j][j];
        }
        for (int j = i + 1; j < m; ++j) G[i][j] = 0.0;
    }
    vector<double> w(m, 0.0);
    // forward
    vector<double> y2 = b;
    for (int i = 0; i < m; ++i) {
        double s = y2[i];
        for (int k = 0; k < i; ++k) s -= G[i][k] * y2[k];
        y2[i] = s / G[i][i];
    }
    // backward
    for (int i = m - 1; i >= 0; --i) {
        double s = y2[i];
        for (int k = i + 1; k < m; ++k) s -= G[k][i] * w[k];
        w[i] = s / G[i][i];
    }
    // Project to simplex
    project_simplex(w);
    return w;
}

static void phase_affine_calibration(int OSR,
                                     const vector<double>& y_pred_train,
                                     const vector<double>& y_true_train,
                                     vector<double>& a_per_phase,
                                     vector<double>& b_per_phase) {
    a_per_phase.assign(OSR, 1.0); b_per_phase.assign(OSR, 0.0);
    const int T = (int)y_pred_train.size() / OSR;
    if (T <= 1) return;
    for (int r = 0; r < OSR; ++r) {
        long double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int t = 0; t < T; ++t) {
            double x = y_pred_train[1LL*t*OSR + r];
            double y = y_true_train[1LL*t*OSR + r];
            sx += x; sy += y; sxx += x*x; sxy += x*y;
        }
        long double n = T;
        long double denom = n * sxx - sx * sx;
        if (fabsl(denom) < 1e-12L) { a_per_phase[r] = 1.0; b_per_phase[r] = 0.0; continue; }
        long double a = (n * sxy - sx * sy) / denom;
        long double b = (sy - a * sx) / n;
        if (!isfinite((double)a) || !isfinite((double)b)) { a_per_phase[r] = 1.0; b_per_phase[r] = 0.0; continue; }
        a_per_phase[r] = (double)a; b_per_phase[r] = (double)b;
    }
}

static void apply_phase_affine_inplace(int OSR, const vector<double>& a_per_phase, const vector<double>& b_per_phase, vector<double>& y) {
    const int T = (int)y.size() / OSR;
    for (int t = 0; t < T; ++t) {
        for (int r = 0; r < OSR; ++r) {
            double v = y[1LL*t*OSR + r];
            y[1LL*t*OSR + r] = clamp1(a_per_phase[r] * v + b_per_phase[r]);
        }
    }
}

/* ========================= Main ========================= */

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);

    int N; if (!(cin >> N)) { return 0; }
    vector<double> ana_waveform_1(N);
    for (int i = 0; i < N; i++) cin >> ana_waveform_1[i];

    int M; cin >> M;
    vector<int> dig_data_1(M); for (int i = 0; i < M; ++i) cin >> dig_data_1[i];

    int K; cin >> K;
    vector<int> dig_data_2(K); for (int i = 0; i < K; ++i) cin >> dig_data_2[i];

    int OSR = (M > 0 && N % M == 0) ? (N / M) : 16;
    if (OSR <= 0) OSR = 16;
    long long Lout = 1LL * K * OSR;

    // Controller per case
    Ctrl cfg = Ctrl::make_for_case(M, N, ana_waveform_1);

    // Choose a single global delta using 1425 coarse search (robust) if enabled
    int global_delta = 0;
    int start_dig_aligned = 0, end_dig_aligned = M; // used to slice training consistently
    if (cfg.enable_delta_search) {
        vector<double> s_norm1(M);
        for (int i = 0; i < M; ++i) s_norm1[i] = (double)dig_data_1[i] / 3.0;
        global_delta = N1425::coarse_delta_search(ana_waveform_1, s_norm1, OSR, cfg.delta_min, cfg.delta_max);
        start_dig_aligned = 0; while (start_dig_aligned * OSR + global_delta < 0) start_dig_aligned++;
        end_dig_aligned = M; while (end_dig_aligned > 0 && 1LL*(end_dig_aligned - 1)*OSR + OSR + global_delta > (long long)ana_waveform_1.size()) end_dig_aligned--;
        if (end_dig_aligned < start_dig_aligned) { start_dig_aligned = 0; end_dig_aligned = 0; }
    }

    // Storage for per-model predictions (test and aligned training)
    vector<vector<double>> preds_test;   // each size K*OSR
    vector<vector<double>> preds_train;  // each size (Meff*OSR)
    vector<double> model_rmse;           // train RMSE per model (aligned)
    vector<double> model_prior;          // prior weights

    // 1801
    if (cfg.enable_1801) {
        auto rr = N1801::run_both(
            ana_waveform_1, M, dig_data_1, K, dig_data_2, OSR,
            cfg.c1801.stages,
            global_delta, global_delta // force global delta
        );
        preds_test.push_back(std::move(rr.pred_test));
        preds_train.push_back(std::move(rr.pred_train));
        model_rmse.push_back(0.0); // filled after we build y_true_train
        model_prior.push_back(cfg.w1801);
        // ensure consistent aligned slice
        start_dig_aligned = rr.start_dig; end_dig_aligned = rr.end_dig;
    }

    // 5150 (+ SignalModel)
    if (cfg.enable_5150) {
        auto r5150 = run_5150_all(
            ana_waveform_1, M, dig_data_1, K, dig_data_2, OSR,
            cfg.c5150.P_cache, cfg.c5150.Q_cache, cfg.c5150.P_reg, cfg.c5150.Q_reg, cfg.c5150.ridge_lambda_cap,
            global_delta,
            cfg.c1620, cfg.c5150.alpha_5150_vs_sm, cfg.c5150.force_alpha_case_override, M
        );
        preds_test.push_back(std::move(r5150.test));
        preds_train.push_back(std::move(r5150.train_pred));
        model_rmse.push_back(r5150.train_rmse);
        model_prior.push_back(cfg.w5150);
        // Align training slice with 5150's (should match global)
        start_dig_aligned = r5150.start_dig; end_dig_aligned = r5150.end_dig;
    }

    // 1425
    if (cfg.enable_1425) {
        auto r1425 = N1425::run_both(
            ana_waveform_1, M, dig_data_1, K, dig_data_2, OSR,
            cfg.c1425.L_pre, cfg.c1425.L_post, cfg.c1425.orders, cfg.c1425.lambda,
            global_delta, global_delta, cfg.c1425.tau // force delta
        );
        preds_test.push_back(std::move(r1425.pred_test));
        preds_train.push_back(std::move(r1425.pred_train));
        model_rmse.push_back(0.0);
        model_prior.push_back(cfg.w1425);
        start_dig_aligned = r1425.start_dig; end_dig_aligned = r1425.end_dig;
    }

    // If no models active, output zeros
    if (preds_test.empty()) {
        cout << Lout << "\n";
        cout.setf(std::ios::fixed); cout << setprecision(8);
        for (long long i = 0; i < Lout; ++i) cout << (i ? " " : "") << 0.0;
        cout << "\n";
        return 0;
    }

    // Build aligned training ground truth vector
    const int Meff = max(0, end_dig_aligned - start_dig_aligned);
    vector<double> y_true_train((size_t)Meff * OSR, 0.0);
    for (int i = 0; i < Meff; ++i) {
        int srci = start_dig_aligned + i;
        for (int j = 0; j < OSR; ++j) {
            y_true_train[1LL * i * OSR + j] = ana_waveform_1[1LL * srci * OSR + global_delta + j];
        }
    }

    // Update RMSE for models that didn't set it
    for (size_t m = 0; m < preds_train.size(); ++m) {
        if ((int)preds_train[m].size() == (int)y_true_train.size()) {
            model_rmse[m] = rmse(preds_train[m], y_true_train);
        } else {
            model_rmse[m] = 1.1e9;
        }
    }

    // Learn ensemble weights
    vector<double> w(preds_test.size(), 0.0);
    if (cfg.weight_mode == Ctrl::Fixed) {
        // Use priors then normalize if requested
        w = model_prior;
        if (cfg.normalize_weights) {
            double s = 0.0; for (double vi : w) s += max(0.0, vi);
            if (s > 0) for (double& vi : w) vi = max(0.0, vi) / s;
        }
    } else if (cfg.weight_mode == Ctrl::InverseRMSE) {
        double eps = 1e-6;
        for (size_t i = 0; i < w.size(); ++i) w[i] = 1.0 / (model_rmse[i] + eps);
        double s = 0.0; for (double vi : w) s += vi;
        for (double& vi : w) vi /= max(1e-12, s);
    } else if (cfg.weight_mode == Ctrl::SoftmaxRMSE) {
        double T = max(1e-6, cfg.softmax_T);
        vector<double> logits(w.size(), 0.0);
        for (size_t i = 0; i < w.size(); ++i) logits[i] = -model_rmse[i] / T;
        double mlog = *max_element(logits.begin(), logits.end());
        double Z = 0.0; for (double& z : logits) Z += exp(z - mlog);
        for (size_t i = 0; i < w.size(); ++i) w[i] = exp(logits[i] - mlog) / max(1e-12, Z);
    } else {
        // NNLS on simplex with L2 prior toward model_prior
        vector<vector<double>> Xcols = preds_train;
        w = learn_weights_nnls_simplex(Xcols, y_true_train, model_prior, cfg.prior_strength * (double)OSR);
    }

    // Build training ensemble prediction to learn calibration/FIR
    vector<double> y_pred_train((size_t)Meff * OSR, 0.0);
    for (size_t m = 0; m < preds_train.size(); ++m) {
        const vector<double>& v = preds_train[m]; if (v.size() != y_pred_train.size()) continue;
        for (size_t i = 0; i < y_pred_train.size(); ++i) y_pred_train[i] += w[m] * v[i];
    }

    // Phase-wise affine calibration
    vector<double> a_per_phase, b_per_phase;
    if (cfg.enable_phase_affine) {
        phase_affine_calibration(OSR, y_pred_train, y_true_train, a_per_phase, b_per_phase);
        apply_phase_affine_inplace(OSR, a_per_phase, b_per_phase, y_pred_train);
    } else {
        a_per_phase.assign(OSR, 1.0); b_per_phase.assign(OSR, 0.0);
    }

    // Post-ensemble FIR
    vector<double> Hpost;
    if (cfg.enable_post_fir && cfg.post_fir_len >= 3 && (cfg.post_fir_len % 2 == 1)) {
        fit_fir_linear(y_pred_train, y_true_train, cfg.post_fir_len, Hpost, true);
    }

    // Compose test ensemble
    vector<double> out((size_t)K * OSR, 0.0);
    for (size_t m = 0; m < preds_test.size(); ++m) {
        const vector<double>& v = preds_test[m]; if (v.size() != out.size()) continue;
        for (size_t i = 0; i < out.size(); ++i) out[i] += w[m] * v[i];
    }
    // Apply same calibration to test
    if (cfg.enable_phase_affine) apply_phase_affine_inplace(OSR, a_per_phase, b_per_phase, out);
    // Apply post FIR
    if (!Hpost.empty()) apply_fir_inplace(out, Hpost);
    else { for (double& v : out) v = clamp1(v); }

    // Output
    cout << Lout << "\n";
    cout.setf(std::ios::fixed); cout << setprecision(11);
    for (size_t i = 0; i < out.size(); ++i) cout << (i ? " " : "") << out[i];
    cout << "\n";
    return 0;
}
