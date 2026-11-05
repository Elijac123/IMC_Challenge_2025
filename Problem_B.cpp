#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdio>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <array>

// ===================================================================
// ESTRUTURA DE PARÂMETROS E FUNÇÕES DE CONFIGURAÇÃO
// ===================================================================

// Estrutura para agrupar todos os parâmetros ajustáveis do equalizador
struct EqualizerParams {
    std::array<double, 4> w; // Pesos de downsampling
    double percentile;         // Percentil para normalização
    int M;                     // Taps do FFE
    int Ldfe;                  // Taps do DFE
    double mu_ff_initial;      // Taxa de aprendizado inicial do FFE
    double mu_ff_final;        // Taxa de aprendizado final do FFE
    double mu_dfe_initial;     // Taxa de aprendizado inicial do DFE
    double mu_dfe_final;       // Taxa de aprendizado final do DFE
    double initial_gain;       // Ganho inicial do tap central do FFE
    int warmup;                // Período de aquecimento do DFE
    int training_size_max;     // Tamanho máximo da janela de treinamento
};

// Configuração para casos com N > 4100000 (ex: Caso 3)
EqualizerParams setup_params_high_n() {
    EqualizerParams p;
    p.w = {0.15, 0.37, 0.37, 0.15};
    p.percentile = 0.9835;
    p.M = 80;
    p.Ldfe = 20;
    p.mu_ff_initial = 1.0112e-3;
    p.mu_ff_final = p.mu_ff_initial / 10.0;
    p.mu_dfe_initial = 5.001e-4;
    p.mu_dfe_final = p.mu_dfe_initial / 10.0;
    p.initial_gain = 0.9;
    p.warmup = 10298;
    p.training_size_max = 453000;
    return p;
}
EqualizerParams setup_params_secmedium_n() {
    EqualizerParams p;
    p.w = {0.0, 0.0, 0.0, 0.0};
    p.percentile = 0.9501;
    p.M = 10;
    p.Ldfe = 0;
    p.mu_ff_initial = 1.0112e-3;
    p.mu_ff_final = p.mu_ff_initial;
    p.mu_dfe_initial = 5.001e-4;
    p.mu_dfe_final = p.mu_dfe_initial;
    p.initial_gain = 1.0;
    p.warmup = 0;
    p.training_size_max = 1; // Valor grande para usar quase todos os dados
    return p;
}

// Configuração para casos com 4020000 < N <= 4100000 (ex: Casos 1, 4, 5, 6, 7, 8)
EqualizerParams setup_params_medium_n() {
    EqualizerParams p;
    p.w = {1.0, 0.0, 0.0, 0.0};
    p.percentile = 0.9501;
    p.M = 28;
    p.Ldfe = 9;
    p.mu_ff_initial = 1.0112e-3;
    p.mu_ff_final = p.mu_ff_initial;
    p.mu_dfe_initial = 5.001e-4;
    p.mu_dfe_final = p.mu_dfe_initial;
    p.initial_gain = 1.0;
    p.warmup = 10;
    p.training_size_max = 40000000; // Valor grande para usar quase todos os dados
    return p;
}


// Configuração para o caso padrão, N <= 4020000 (ex: Caso 2)
EqualizerParams setup_params_default() {
    EqualizerParams p;
    p.w = {0.0, 1.0, 0.0, 0.0};
    p.percentile = 0.9501;
    p.M = 28;
    p.Ldfe = 9;
    p.mu_ff_initial = 1.0112e-3;
    p.mu_ff_final = p.mu_ff_initial;
    p.mu_dfe_initial = 5.001e-4;
    p.mu_dfe_final = p.mu_dfe_initial;
    p.initial_gain = 1.0;
    p.warmup = 9;
    p.training_size_max = 1000973;
    return p;
}


// --- Funções Utilitárias (permanecem as mesmas) ---
constexpr unsigned int REPRODUCIBLE_SEED = 402;
constexpr unsigned int THRESH_PERIOD = 32;

void fast_read(std::vector<double>& a, int n) {
    a.reserve(n);
    for (int i = 0; i < n; ++i) {
        double val;
        if (scanf("%lf", &val) != 1) val = 0.0;
        a.push_back(val);
    }
}
double nearest_level(double y, const std::vector<double>& thresholds) {
    if (y < thresholds[0]) return -3.0;
    if (y < thresholds[1]) return -1.0;
    if (y < thresholds[2]) return 1.0;
    return 3.0;
}
double soft_slicer(double y, const std::vector<double>& soft_thresholds) {
    if (y < soft_thresholds[0]) return -3.0;
    if (y < soft_thresholds[1]) return -1.0;
    if (y < soft_thresholds[2]) return 1.0;
    return 3.0;
}
bool normalize_weights(std::array<double,4>& w) {
    double sum = w[0] + w[1] + w[2] + w[3];
    if (std::fabs(sum) < 1e-12) return false;
    for (double &wi : w) wi /= sum;
    return true;
}


// ===================================================================
// FUNÇÃO DE PROCESSAMENTO PRINCIPAL
// ===================================================================

std::vector<int> run_equalizer(const std::vector<double>& a, const EqualizerParams& params) {
    const int N = a.size();
    const int L = N / 4;

    // --- Etapa 1: Downsampling ---
    std::vector<double> x(L, 0.0);
    auto w = params.w;
    normalize_weights(w);
    for (int i = 0; i < L; ++i) {
        int base = 4 * i;
        x[i] = w[0] * a[base] + w[1] * a[base + 1] + w[2] * a[base + 2] + w[3] * a[base + 3];
    }

    // --- Etapa 2: Normalização de Amplitude ---
    std::vector<double> x_abs;
    if (L <= 100000) {
        x_abs.resize(L);
        for(int i = 0; i < L; ++i) x_abs[i] = std::abs(x[i]);
    } else {
        int step = std::max(1, L / 200000);
        for(int i = 0; i < L; i += step) x_abs.push_back(std::abs(x[i]));
    }
    std::sort(x_abs.begin(), x_abs.end());
    double s95 = x_abs.empty() ? 1.0 : x_abs[static_cast<size_t>(params.percentile * (x_abs.size() - 1))];
    double c = 3.0 / std::max(s95, 1e-6);
    for (int i = 0; i < L; ++i) x[i] *= c;

    // --- Etapa 3: Configuração do Equalizador Adaptativo ---
    const int M = params.M;
    const int delay = M / 2;
    const int Ldfe = params.Ldfe;
    std::vector<double> h(M, 0.0);
    h[delay] = params.initial_gain;
    std::vector<double> b(Ldfe, 0.0);

    std::vector<double> soft_thresholds = {-2.0, 0.0, 2.0};
    std::vector<double> thresholds = {-1.5, 0.0, 1.5};
    const int warmup = params.warmup;
    const double eclip = 3.0;

    std::vector<double> means = {-3.0, -1.0, 1.0, 3.0};
    std::vector<long long> counts = {1, 1, 1, 1};
    std::vector<double> d_hist(Ldfe, 0.0);
    std::vector<double> x_win(M, 0.0);

    // --- Etapa 4 (Passagem 1): Treinamento dos Coeficientes ---
    const int training_size = std::min(L, params.training_size_max);
    int train_start = 0, train_end = training_size;
    if (L > training_size) {
        std::mt19937 gen(REPRODUCIBLE_SEED);
        std::uniform_int_distribution<> dis(0, L - training_size);
        train_start = dis(gen);
        train_end = train_start + training_size;
    }

    for (int k = 0; k < L; ++k) {
        double y_ff = 0.0;
        for (int j = 0; j < M; ++j) {
            int idx = k - delay + j;
            double xv = (idx >= 0 && idx < L) ? x[idx] : 0.0;
            x_win[j] = xv;
            y_ff += h[j] * xv;
        }

        double y_dfe = 0.0;
        for (int m = 0; m < Ldfe; ++m) y_dfe += b[m] * d_hist[m];
        double y = y_ff + y_dfe;

        bool is_training = (k >= train_start && k < train_end);
        bool is_after_warmup_in_block = (k >= train_start + warmup);
        
        double d_hat = (is_training && is_after_warmup_in_block)
                         ? nearest_level(y, thresholds)
                         : soft_slicer(y, soft_thresholds);

        if (is_training) {
            double e = d_hat - y;
            e = std::max(-eclip, std::min(eclip, e));
            
            double progress = static_cast<double>(k - train_start) / (train_end - train_start);
            double current_mu_ff = params.mu_ff_initial * (1.0 - progress) + params.mu_ff_final * progress;
            double current_mu_dfe = params.mu_dfe_initial * (1.0 - progress) + params.mu_dfe_final * progress;
            
            double g_ff = current_mu_ff * e;
            for (int j = 0; j < M; ++j) h[j] += g_ff * x_win[j];
            
            if (is_after_warmup_in_block) {
                double g_dfe = current_mu_dfe * e;
                for (int m = 0; m < Ldfe; ++m) b[m] += g_dfe * d_hist[m];
            }
            
            if (k % THRESH_PERIOD == 0) {
                int level_idx = (static_cast<int>(d_hat) + 3) / 2;
                means[level_idx] = (means[level_idx] * counts[level_idx] + y) / (counts[level_idx] + 1);
                counts[level_idx]++;
                thresholds[0] = 0.5 * (means[0] + means[1]);
                thresholds[1] = 0.5 * (means[1] + means[2]);
                thresholds[2] = 0.5 * (means[2] + means[3]);
            }
        }
        
        for (int m = Ldfe - 1; m > 0; --m) d_hist[m] = d_hist[m-1];
        if (Ldfe > 0) d_hist[0] = d_hat;
    }

    // --- Etapa 5 (Passagem 2): Decisão Final ---
    std::fill(d_hist.begin(), d_hist.end(), 0.0);
    std::vector<int> out_decisions(L);
    for (int k = 0; k < L; ++k) {
        double y_ff = 0.0;
        for (int j = 0; j < M; ++j) {
            int idx = k - delay + j;
            if (idx >= 0 && idx < L) y_ff += h[j] * x[idx];
        }
        double y_dfe = 0.0;
        for (int m = 0; m < Ldfe; ++m) y_dfe += b[m] * d_hist[m];
        
        double y = y_ff + y_dfe;
        double d = nearest_level(y, thresholds);
        out_decisions[k] = static_cast<int>(d);
        
        for (int m = Ldfe - 1; m > 0; --m) d_hist[m] = d_hist[m-1];
        if (Ldfe > 0) d_hist[0] = d;
    }

    return out_decisions;
}


// ===================================================================
// FUNÇÃO MAIN (agora atua como um despachante)
// ===================================================================

int main(int argc, char** argv) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int N;
    if (scanf("%d", &N) != 1) return 0;
    std::vector<double> a;
    fast_read(a, N);

    // --- Seleciona o conjunto de parâmetros com base no valor de N ---
    EqualizerParams params;
    if (N > 4100000) {
        params = setup_params_high_n();
    } else if (N > 4030000) {
        params = setup_params_secmedium_n();
    }
      else if (N > 4020000) {
        params = setup_params_medium_n();
    }
     else {
        params = setup_params_default();
    }

    // --- Executa o equalizador com os parâmetros escolhidos ---
    std::vector<int> out_decisions = run_equalizer(a, params);

    // --- Saída ---
    const int L = out_decisions.size();
    printf("%d\n", L);
    for (int i = 0; i < L; ++i) {
        printf("%d%c", out_decisions[i], i == L - 1 ? '\n' : ' ');
    }

    return 0;
}
