/* Monte Carlo Metropolis de grafeno, protocolo Fasolino et al. (2007).
 * N = 240, T <= 300 K. Potencial Tersoff C (Lindsay–Broido).
 * Réplicas independientes en OpenMP; deja nproc-2 hilos para el sistema.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define A_CC 1.42
#define KB_EV 8.617333262145e-5
#define KAPPA 1.1
#define MAX_N 240
#define MAX_NN 20
#define NMAX_Q 10
#define RDF_BINS 350
#define RDF_MAX 3.5
#define RAM_RESERVE_BYTES (2L * 1024L * 1024L * 1024L)

/* Tersoff C, Lindsay–Broido 2010 */
static const double TER_A = 1393.6;
static const double TER_B = 430.0;
static const double TER_L1 = 3.4879;
static const double TER_L2 = 2.2119;
static const double TER_BETA = 1.5724e-7;
static const double TER_N = 0.72751;
static const double TER_C = 38049.0;
static const double TER_D = 4.3484;
static const double TER_H = -0.93;
static const double TER_R = 1.80;
static const double TER_S = 2.10;

typedef struct {
    uint64_t s;
} Rng;

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(Rng *r, uint64_t seed) {
    r->s = splitmix64(&seed);
    if (r->s == 0) r->s = 0xA5A5A5A5A5A5A5A5ULL;
}

static uint64_t rng_u64(Rng *r) {
    uint64_t x = r->s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    r->s = x;
    return x;
}

static double rng_uni(Rng *r) {
    return (rng_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_sym(Rng *r) { return 2.0 * rng_uni(r) - 1.0; }

static double cutoff_fc(double r) {
    if (r < TER_R) return 1.0;
    if (r >= TER_S) return 0.0;
    double x = M_PI * (r - TER_R) / (TER_S - TER_R);
    return 0.5 + 0.5 * cos(x);
}

static double g_theta(double cost) {
    double c2 = TER_C * TER_C;
    double d2 = TER_D * TER_D;
    double dh = TER_H - cost;
    return 1.0 + c2 / d2 - c2 / (d2 + dh * dh);
}

static void min_image(double *dx, double *dy, double lx, double ly) {
    *dx -= lx * nearbyint(*dx / lx);
    *dy -= ly * nearbyint(*dy / ly);
}

static int build_neigh(const double *x, const double *y, const double *z, int n,
                       int i, double lx, double ly, int *nb) {
    int m = 0;
    const double s2 = TER_S * TER_S;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
        min_image(&dx, &dy, lx, ly);
        if (dx * dx + dy * dy + dz * dz < s2) {
            nb[m++] = j;
            if (m >= MAX_NN) break;
        }
    }
    return m;
}

static double pair_term_nb(const double *x, const double *y, const double *z,
                           int i, int j, const int *nb, int nn, double lx,
                           double ly) {
    double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
    min_image(&dx, &dy, lx, ly);
    double rij = sqrt(dx * dx + dy * dy + dz * dz);
    double fc = cutoff_fc(rij);
    if (fc == 0.0 || rij < 1e-14) return 0.0;
    double zeta = 0.0;
    for (int a = 0; a < nn; a++) {
        int k = nb[a];
        if (k == j) continue;
        double dxk = x[k] - x[i], dyk = y[k] - y[i], dzk = z[k] - z[i];
        min_image(&dxk, &dyk, lx, ly);
        double rik = sqrt(dxk * dxk + dyk * dyk + dzk * dzk);
        double fck = cutoff_fc(rik);
        if (fck == 0.0 || rik < 1e-14) continue;
        double cost = (dx * dxk + dy * dyk + dz * dzk) / (rij * rik);
        if (cost > 1.0) cost = 1.0;
        if (cost < -1.0) cost = -1.0;
        zeta += fck * g_theta(cost);
    }
    double zn = (zeta > 0.0) ? pow(zeta, TER_N) : 0.0;
    double b = pow(1.0 + pow(TER_BETA, TER_N) * zn, -0.5 / TER_N);
    double fR = TER_A * exp(-TER_L1 * rij);
    double fA = -TER_B * exp(-TER_L2 * rij);
    return fc * (fR + b * fA);
}

double tersoff_energy(const double *x, const double *y, const double *z,
                      int n, double lx, double ly) {
    double e = 0.0;
    int nb[MAX_NN];
    for (int i = 0; i < n; i++) {
        int nn = build_neigh(x, y, z, n, i, lx, ly, nb);
        for (int a = 0; a < nn; a++)
            e += 0.5 * pair_term_nb(x, y, z, i, nb[a], nb, nn, lx, ly);
    }
    return e;
}

/* Energía que cambia al mover el átomo i (ver tests de consistencia). */
static double local_energy(const double *x, const double *y, const double *z,
                           int n, int i, double lx, double ly) {
    double e = 0.0;
    int nbi[MAX_NN], nbj[MAX_NN];
    int ni = build_neigh(x, y, z, n, i, lx, ly, nbi);
    for (int a = 0; a < ni; a++) {
        int j = nbi[a];
        e += 0.5 * pair_term_nb(x, y, z, i, j, nbi, ni, lx, ly);
        int nj = build_neigh(x, y, z, n, j, lx, ly, nbj);
        e += 0.5 * pair_term_nb(x, y, z, j, i, nbj, nj, lx, ly);
        for (int b = 0; b < nj; b++) {
            int k = nbj[b];
            if (k == i) continue;
            e += 0.5 * pair_term_nb(x, y, z, j, k, nbj, nj, lx, ly);
        }
    }
    return e;
}

static void make_lattice(double *x, double *y, double *z, int p, int q,
                         double *lx, double *ly, int *n) {
    double a1 = A_CC * sqrt(3.0);
    *lx = p * a1;
    *ly = q * 3.0 * A_CC;
    *n = 4 * p * q;
    int idx = 0;
    double bx[4] = {0.0, 0.0, 0.5 * a1, 0.5 * a1};
    double by[4] = {0.0, A_CC, 1.5 * A_CC, 2.5 * A_CC};
    for (int iy = 0; iy < q; iy++) {
        for (int ix = 0; ix < p; ix++) {
            for (int b = 0; b < 4; b++) {
                x[idx] = bx[b] + ix * a1;
                y[idx] = by[b] + iy * 3.0 * A_CC;
                z[idx] = 0.0;
                idx++;
            }
        }
    }
}

static double wrap(double v, double L) {
    v -= L * floor(v / L);
    return v;
}

typedef struct {
    double q;
    double g_over_n;
} GqPoint;

static long mem_available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char key[64], unit[32];
    long val = 0, avail = 0;
    while (fscanf(f, "%63s %ld %31s", key, &val, unit) == 3) {
        if (strcmp(key, "MemAvailable:") == 0) {
            avail = val * 1024L;
            break;
        }
    }
    fclose(f);
    return avail;
}

static int sample_every_from_ram(int n_prod, int n_replicas, int requested) {
    if (requested > 0) return requested;
    if (n_prod <= 0 || n_replicas <= 0) return 100;
    long avail = mem_available_bytes();
    long usable = avail > RAM_RESERVE_BYTES ? avail - RAM_RESERVE_BYTES : avail / 2;
    long per = usable / n_replicas;
    /* 3 coordenadas + caja por muestra */
    long bytes_each = (long)MAX_N * 3L * (long)sizeof(double) + 2L * (long)sizeof(double);
    long cap = (per - 8L * 1024L * 1024L) / bytes_each;
    /* tope global 8 GiB de muestras para no ahogar al escritorio */
    long max_total = 8L * 1024L * 1024L * 1024L;
    long cap_tot = max_total / ((long)n_replicas * bytes_each);
    if (cap_tot < cap) cap = cap_tot;
    if (cap < 2000) cap = 2000;
    int every = (int)((n_prod + cap - 1) / cap);
    if (every < 1) every = 1;
    if (every > 100) every = 100;
    return every;
}

typedef struct {
    int n;
    double lx, ly;
    double x[MAX_N], y[MAX_N], z[MAX_N];
    double r2[MAX_N][MAX_N];
    int nn[MAX_N];
    int neigh[MAX_N][MAX_NN];
    int ntrace, ntrace_cap;
    int *tr_step;
    double *tr_e, *tr_h, *tr_lx, *tr_ly, *tr_accm, *tr_acca;
    int nsamp, nsamp_cap;
    double *sx, *sy, *sz, *slx, *sly;
} Workspace;

static void ws_rebuild_row(Workspace *w, int i) {
    const double s2 = TER_S * TER_S;
    int n = w->n;
    w->nn[i] = 0;
    for (int j = 0; j < n; j++) {
        if (j == i) {
            w->r2[i][i] = 0.0;
            continue;
        }
        double dx = w->x[j] - w->x[i], dy = w->y[j] - w->y[i], dz = w->z[j] - w->z[i];
        min_image(&dx, &dy, w->lx, w->ly);
        double r2 = dx * dx + dy * dy + dz * dz;
        w->r2[i][j] = w->r2[j][i] = r2;
        if (r2 < s2 && w->nn[i] < MAX_NN) w->neigh[i][w->nn[i]++] = j;
    }
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        int has = 0, pos = -1;
        for (int a = 0; a < w->nn[j]; a++) {
            if (w->neigh[j][a] == i) {
                has = 1;
                pos = a;
                break;
            }
        }
        int should = (w->r2[i][j] < s2);
        if (should && !has && w->nn[j] < MAX_NN) w->neigh[j][w->nn[j]++] = i;
        if (!should && has) w->neigh[j][pos] = w->neigh[j][--w->nn[j]];
    }
}

static void ws_rebuild_all(Workspace *w) {
    memset(w->nn, 0, sizeof(w->nn));
    for (int i = 0; i < w->n; i++) ws_rebuild_row(w, i);
}

static double local_energy_ws(const Workspace *w, int i) {
    double e = 0.0;
    int ni = w->nn[i];
    const int *nbi = w->neigh[i];
    for (int a = 0; a < ni; a++) {
        int j = nbi[a];
        e += 0.5 * pair_term_nb(w->x, w->y, w->z, i, j, nbi, ni, w->lx, w->ly);
        int nj = w->nn[j];
        const int *nbj = w->neigh[j];
        e += 0.5 * pair_term_nb(w->x, w->y, w->z, j, i, nbj, nj, w->lx, w->ly);
        for (int b = 0; b < nj; b++) {
            int k = nbj[b];
            if (k == i) continue;
            e += 0.5 * pair_term_nb(w->x, w->y, w->z, j, k, nbj, nj, w->lx, w->ly);
        }
    }
    return e;
}

static void accumulate_observables(const double *x, const double *y, const double *z,
                                   int n, double lx, double ly,
                                   double *gq_sum, int *gq_count, int nmax,
                                   double *rdf, double *h_rms_acc, int *nsamp) {
    double zmean = 0.0;
    for (int i = 0; i < n; i++) zmean += z[i];
    zmean /= n;
    double h2 = 0.0;
    double h[MAX_N];
    for (int i = 0; i < n; i++) {
        h[i] = z[i] - zmean;
        h2 += h[i] * h[i];
    }
    *h_rms_acc += sqrt(h2 / n);
    (*nsamp)++;

    int idx = 0;
    for (int nx = -nmax; nx <= nmax; nx++) {
        for (int ny = -nmax; ny <= nmax; ny++) {
            if (nx == 0 && ny == 0) {
                idx++;
                continue;
            }
            double qx = 2.0 * M_PI * nx / lx;
            double qy = 2.0 * M_PI * ny / ly;
            double q2 = qx * qx + qy * qy;
            double hr = 0.0, hi = 0.0;
            for (int j = 0; j < n; j++) {
                double ph = qx * x[j] + qy * y[j];
                hr += h[j] * cos(ph);
                hi -= h[j] * sin(ph);
            }
            double g = q2 * (hr * hr + hi * hi);
            gq_sum[idx] += g / n;
            gq_count[idx] += 1;
            idx++;
        }
    }

    double dr = RDF_MAX / RDF_BINS;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
            min_image(&dx, &dy, lx, ly);
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r >= RDF_MAX) continue;
            int b = (int)(r / dr);
            if (b >= 0 && b < RDF_BINS) rdf[b] += 2.0;
        }
    }
}

static void run_replica(int replica, int p, int q, double T, int n_eq, int n_prod,
                        int sample_every, uint64_t seed, const char *outdir) {
    Workspace *w = (Workspace *)calloc(1, sizeof(Workspace));
    if (!w) {
        fprintf(stderr, "sin RAM para la réplica %d\n", replica);
        return;
    }
    make_lattice(w->x, w->y, w->z, p, q, &w->lx, &w->ly, &w->n);
    int n = w->n;
    if (n > MAX_N) {
        fprintf(stderr, "N=%d > MAX_N\n", n);
        free(w);
        return;
    }
    ws_rebuild_all(w);

    int ntrace_cap = (n_eq + n_prod) / 200 + 8;
    int nsamp_cap = n_prod / sample_every + 8;
    w->ntrace_cap = ntrace_cap;
    w->nsamp_cap = nsamp_cap;
    w->tr_step = calloc((size_t)ntrace_cap, sizeof(int));
    w->tr_e = calloc((size_t)ntrace_cap, sizeof(double));
    w->tr_h = calloc((size_t)ntrace_cap, sizeof(double));
    w->tr_lx = calloc((size_t)ntrace_cap, sizeof(double));
    w->tr_ly = calloc((size_t)ntrace_cap, sizeof(double));
    w->tr_accm = calloc((size_t)ntrace_cap, sizeof(double));
    w->tr_acca = calloc((size_t)ntrace_cap, sizeof(double));
    w->sx = malloc((size_t)nsamp_cap * (size_t)n * sizeof(double));
    w->sy = malloc((size_t)nsamp_cap * (size_t)n * sizeof(double));
    w->sz = malloc((size_t)nsamp_cap * (size_t)n * sizeof(double));
    w->slx = malloc((size_t)nsamp_cap * sizeof(double));
    w->sly = malloc((size_t)nsamp_cap * sizeof(double));
    if (!w->sx || !w->sy || !w->sz || !w->tr_e) {
        fprintf(stderr, "réplica %d: no se pudo reservar el buffer RAM\n", replica);
        free(w);
        return;
    }
    /* Comprometer páginas: calloc/memset-cero los elide el compilador. */
    {
        size_t xyz = (size_t)nsamp_cap * (size_t)n * sizeof(double);
        volatile char *px = (volatile char *)w->sx;
        volatile char *py = (volatile char *)w->sy;
        volatile char *pz = (volatile char *)w->sz;
        for (size_t i = 0; i < xyz; i += 4096) {
            px[i] = 1;
            py[i] = 1;
            pz[i] = 1;
        }
        if (xyz) {
            px[xyz - 1] = 1;
            py[xyz - 1] = 1;
            pz[xyz - 1] = 1;
        }
    }

    Rng rng;
    rng_seed(&rng, seed + (uint64_t)replica * 0xD1B54A32D192ED03ULL);

    double beta = 1.0 / (KB_EV * T);
    double step = 0.04;
    double area_step = 0.002;
    double energy = tersoff_energy(w->x, w->y, w->z, n, w->lx, w->ly);

    int nmax = NMAX_Q;
    int nmodes = (2 * nmax + 1) * (2 * nmax + 1);
    double *gq_sum = calloc((size_t)nmodes, sizeof(double));
    int *gq_count = calloc((size_t)nmodes, sizeof(int));
    double rdf[RDF_BINS];
    memset(rdf, 0, sizeof(rdf));
    double h_rms_acc = 0.0;
    int nsamp = 0;

    long acc_move = 0, try_move = 0, acc_area = 0, try_area = 0;
    double h_rms_eq = 0.0;

    int total = n_eq + n_prod;
    for (int istep = 1; istep <= total; istep++) {
        for (int att = 0; att < n; att++) {
            int i = (int)(rng_uni(&rng) * n);
            if (i >= n) i = n - 1;
            double e_old = local_energy_ws(w, i);
            double ox = w->x[i], oy = w->y[i], oz = w->z[i];
            w->x[i] = wrap(ox + step * rng_sym(&rng), w->lx);
            w->y[i] = wrap(oy + step * rng_sym(&rng), w->ly);
            w->z[i] = oz + step * rng_sym(&rng);
            ws_rebuild_row(w, i);
            double e_new = local_energy_ws(w, i);
            double dE = e_new - e_old;
            try_move++;
            int ok = (dE <= 0.0) || (rng_uni(&rng) < exp(-beta * dE));
            if (ok) {
                energy += dE;
                acc_move++;
            } else {
                w->x[i] = ox;
                w->y[i] = oy;
                w->z[i] = oz;
                ws_rebuild_row(w, i);
            }
        }

        if (istep % 5 == 0) {
            double factor = exp(area_step * rng_sym(&rng));
            double lx_n = w->lx * factor, ly_n = w->ly * factor;
            double e_old = energy;
            double xs[MAX_N], ys[MAX_N];
            for (int i = 0; i < n; i++) {
                xs[i] = w->x[i];
                ys[i] = w->y[i];
                w->x[i] *= factor;
                w->y[i] *= factor;
            }
            double e_new = tersoff_energy(w->x, w->y, w->z, n, lx_n, ly_n);
            double dE = e_new - e_old;
            double dlogA = 2.0 * log(factor);
            double acc_p = exp(-beta * dE + n * dlogA);
            try_area++;
            if (rng_uni(&rng) < acc_p) {
                w->lx = lx_n;
                w->ly = ly_n;
                energy = e_new;
                acc_area++;
                ws_rebuild_all(w);
            } else {
                for (int i = 0; i < n; i++) {
                    w->x[i] = xs[i];
                    w->y[i] = ys[i];
                }
            }
        }

        if (istep == n_eq) {
            double zmean = 0.0;
            for (int i = 0; i < n; i++) zmean += w->z[i];
            zmean /= n;
            double h2 = 0.0;
            for (int i = 0; i < n; i++) {
                double hi = w->z[i] - zmean;
                h2 += hi * hi;
            }
            h_rms_eq = sqrt(h2 / n);
        }

        if (istep > n_eq && (istep - n_eq) % sample_every == 0 && nsamp < w->nsamp_cap) {
            size_t off = (size_t)nsamp * (size_t)n;
            memcpy(w->sx + off, w->x, (size_t)n * sizeof(double));
            memcpy(w->sy + off, w->y, (size_t)n * sizeof(double));
            memcpy(w->sz + off, w->z, (size_t)n * sizeof(double));
            w->slx[nsamp] = w->lx;
            w->sly[nsamp] = w->ly;
            accumulate_observables(w->x, w->y, w->z, n, w->lx, w->ly, gq_sum,
                                   gq_count, nmax, rdf, &h_rms_acc, &nsamp);
            /* nsamp ya incrementado en accumulate */
        }

        if ((istep % 200 == 0 || istep == total) && w->ntrace < w->ntrace_cap) {
            double zmean = 0.0;
            for (int i = 0; i < n; i++) zmean += w->z[i];
            zmean /= n;
            double h2 = 0.0;
            for (int i = 0; i < n; i++) {
                double hi = w->z[i] - zmean;
                h2 += hi * hi;
            }
            int t = w->ntrace++;
            w->tr_step[t] = istep;
            w->tr_e[t] = energy;
            w->tr_h[t] = sqrt(h2 / n);
            w->tr_lx[t] = w->lx;
            w->tr_ly[t] = w->ly;
            w->tr_accm[t] = try_move ? (double)acc_move / try_move : 0.0;
            w->tr_acca[t] = try_area ? (double)acc_area / try_area : 0.0;
        }

        if (istep % 400 == 0 && try_move > 0) {
            double ar = (double)acc_move / try_move;
            if (ar > 0.55) step *= 1.08;
            if (ar < 0.35) step *= 0.92;
            if (step < 0.005) step = 0.005;
            if (step > 0.12) step = 0.12;
        }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/replica_%02d_trace.csv", outdir, replica);
    FILE *trace = fopen(path, "w");
    if (trace) {
        fprintf(trace, "step,energy,h_rms,lx,ly,acc_move,acc_area\n");
        for (int t = 0; t < w->ntrace; t++)
            fprintf(trace, "%d,%.8f,%.8f,%.8f,%.8f,%.6f,%.6f\n", w->tr_step[t],
                    w->tr_e[t], w->tr_h[t], w->tr_lx[t], w->tr_ly[t],
                    w->tr_accm[t], w->tr_acca[t]);
        fclose(trace);
    }

    snprintf(path, sizeof(path), "%s/replica_%02d_summary.csv", outdir, replica);
    FILE *sum = fopen(path, "w");
    if (sum) {
        fprintf(sum,
                "replica,N,T,n_eq,n_prod,energy,h_rms,h_rms_eq,lx,ly,S0,"
                "acc_move,acc_area,nsamples,step_amp\n");
        fprintf(sum, "%d,%d,%.2f,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.6f,%d,%.5f\n",
                replica, n, T, n_eq, n_prod, energy,
                nsamp ? h_rms_acc / nsamp : 0.0, h_rms_eq, w->lx, w->ly,
                w->lx * w->ly / n,
                try_move ? (double)acc_move / try_move : 0.0,
                try_area ? (double)acc_area / try_area : 0.0, nsamp, step);
        fclose(sum);
    }

    snprintf(path, sizeof(path), "%s/replica_%02d_gq.csv", outdir, replica);
    FILE *gf = fopen(path, "w");
    if (gf) {
        fprintf(gf, "nx,ny,q,G_over_N\n");
        int idx = 0;
        for (int nx = -nmax; nx <= nmax; nx++) {
            for (int ny = -nmax; ny <= nmax; ny++) {
                if (nx == 0 && ny == 0) {
                    idx++;
                    continue;
                }
                double qx = 2.0 * M_PI * nx / w->lx;
                double qy = 2.0 * M_PI * ny / w->ly;
                double q = sqrt(qx * qx + qy * qy);
                double g = (gq_count[idx] > 0) ? gq_sum[idx] / gq_count[idx] : 0.0;
                fprintf(gf, "%d,%d,%.8f,%.8e\n", nx, ny, q, g);
                idx++;
            }
        }
        fclose(gf);
    }

    snprintf(path, sizeof(path), "%s/replica_%02d_rdf.csv", outdir, replica);
    FILE *rf = fopen(path, "w");
    if (rf) {
        fprintf(rf, "r,g\n");
        double density = n / (w->lx * w->ly);
        double dr = RDF_MAX / RDF_BINS;
        for (int b = 0; b < RDF_BINS; b++) {
            double r = (b + 0.5) * dr;
            double shell = 2.0 * M_PI * r * dr * density;
            double g = (nsamp > 0 && shell > 0.0) ? rdf[b] / (nsamp * n * shell) : 0.0;
            fprintf(rf, "%.6f,%.8f\n", r, g);
        }
        fclose(rf);
    }

    snprintf(path, sizeof(path), "%s/replica_%02d_snapshot.csv", outdir, replica);
    FILE *sf = fopen(path, "w");
    if (sf) {
        fprintf(sf, "x,y,z,lx,ly\n");
        for (int i = 0; i < n; i++)
            fprintf(sf, "%.8f,%.8f,%.8f,%.8f,%.8f\n", w->x[i], w->y[i], w->z[i],
                    w->lx, w->ly);
        fclose(sf);
    }

    free(gq_sum);
    free(gq_count);
    free(w->tr_step);
    free(w->tr_e);
    free(w->tr_h);
    free(w->tr_lx);
    free(w->tr_ly);
    free(w->tr_accm);
    free(w->tr_acca);
    free(w->sx);
    free(w->sy);
    free(w->sz);
    free(w->slx);
    free(w->sly);
    free(w);
    printf("replica %02d  E=%.4f eV  h_rms=%.4f Å  acc=%.3f  samples=%d\n",
           replica, energy, nsamp ? h_rms_acc / nsamp : 0.0,
           try_move ? (double)acc_move / try_move : 0.0, nsamp);
    fflush(stdout);
}

static int self_test(void) {
    double x[MAX_N], y[MAX_N], z[MAX_N];
    double lx, ly;
    int n;
    make_lattice(x, y, z, 10, 6, &lx, &ly, &n);
    double e0 = tersoff_energy(x, y, z, n, lx, ly);
    int i = 7;
    double loc0 = local_energy(x, y, z, n, i, lx, ly);
    x[i] += 0.03;
    y[i] -= 0.02;
    z[i] += 0.04;
    double e1 = tersoff_energy(x, y, z, n, lx, ly);
    double loc1 = local_energy(x, y, z, n, i, lx, ly);
    double d_full = e1 - e0;
    double d_loc = loc1 - loc0;
    double err = fabs(d_full - d_loc);
    printf("self-test  N=%d  Lx=%.5f  Ly=%.5f  E/N=%.6f eV\n", n, lx, ly, e0 / n);
    printf("self-test  dE_full=%.8f  dE_local=%.8f  |diff|=%.3e\n", d_full, d_loc, err);
    if (n != 240) return 1;
    if (fabs(lx - 24.59) > 0.05) return 1;
    if (fabs(ly - 25.56) > 0.05) return 1;
    if (e0 / n > -4.0) return 1;
    if (err > 1e-6) return 1;
    printf("self-test PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    int p = 10, q = 6;
    double T = 300.0;
    int n_eq = 400000;
    int n_prod = 500000;
    int sample_every = 0;
    int n_replicas = 0;
    uint64_t seed = 42;
    const char *outdir = "results/datos";
    int leave_cores = 2;
    int do_self_test = 0;
    int energy_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--T") && i + 1 < argc) T = atof(argv[++i]);
        else if (!strcmp(argv[i], "--n-eq") && i + 1 < argc) n_eq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-prod") && i + 1 < argc) n_prod = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sample-every") && i + 1 < argc)
            sample_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--replicas") && i + 1 < argc)
            n_replicas = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--leave-cores") && i + 1 < argc)
            leave_cores = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--p") && i + 1 < argc) p = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--self-test")) do_self_test = 1;
        else if (!strcmp(argv[i], "--energy-only")) energy_only = 1;
        else if (!strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "Usage: graphene_mc [--T 300] [--n-eq 400000] [--n-prod 500000]\n"
                    "                   [--replicas N] [--leave-cores 2] [--out DIR]\n");
            return 0;
        }
    }

    if (do_self_test) return self_test();
    if (energy_only) {
        double x[MAX_N], y[MAX_N], z[MAX_N], lx, ly;
        int n;
        make_lattice(x, y, z, p, q, &lx, &ly, &n);
        double e = tersoff_energy(x, y, z, n, lx, ly);
        printf("%.12f %.8f %.8f %d\n", e, lx, ly, n);
        return 0;
    }

    if (T > 300.0 + 1e-9) {
        fprintf(stderr, "T máxima permitida: 300 K (pedido de la reimplementación).\n");
        return 1;
    }
    if (4 * p * q != 240 && !(p == 10 && q == 6)) {
        /* se permite N=240 por defecto; otros tamaños solo para tests */
    }

    int nproc = 1;
#ifdef _OPENMP
    nproc = omp_get_num_procs();
#endif
    int nthreads = nproc - leave_cores;
    if (nthreads < 1) nthreads = 1;
    if (n_replicas <= 0) n_replicas = nthreads;
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif

    sample_every = sample_every_from_ram(n_prod, n_replicas, sample_every);
    long avail = mem_available_bytes();
    long usable = avail > RAM_RESERVE_BYTES ? avail - RAM_RESERVE_BYTES : avail / 2;
    long nsamp_est = n_prod / sample_every;
    long bytes_rep = sizeof(Workspace) + nsamp_est * (long)(4 * p * q) * 3 * (long)sizeof(double);
    printf("Grafeno MC  N=%d  T=%.1f K  eq=%d  prod=%d  replicas=%d  hilos=%d (de %d; se dejan %d)\n",
           4 * p * q, T, n_eq, n_prod, n_replicas, nthreads, nproc, leave_cores);
    printf("RAM disponible %.2f GiB; se reservan 2.00 GiB al sistema; ~%.2f GiB para %d réplicas "
           "(muestra cada %d pasos, ~%ld configs/réplica, ~%.0f MiB/réplica)\n",
           avail / 1024.0 / 1024.0 / 1024.0, usable / 1024.0 / 1024.0 / 1024.0,
           n_replicas, sample_every, nsamp_est, bytes_rep / 1024.0 / 1024.0);
    fflush(stdout);

    double t0 = 0.0;
#ifdef _OPENMP
    t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
#endif
    for (int r = 0; r < n_replicas; r++) {
        run_replica(r, p, q, T, n_eq, n_prod, sample_every, seed, outdir);
    }
#ifdef _OPENMP
    printf("tiempo wall = %.1f s\n", omp_get_wtime() - t0);
#endif
    return 0;
}
