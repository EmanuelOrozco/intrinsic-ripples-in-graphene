/* Monte Carlo Metropolis de grafeno, protocolo Fasolino, Los y Katsnelson (2007).
 * Tamaños de la misma familia que la Tabla I: N = 4 p q, con tope N = 8640
 * (p = 60, q = 36, L > 70 Å). T <= 300 K. Tersoff C de Lindsay–Broido.
 * Cada réplica es un proceso. Las listas de celdas dejan cada movimiento en O(1).
 * Heartbeat y checkpoint permiten reanudar si el proceso muere o deja de avanzar.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define A_CC 1.42
#define KB_EV 8.617333262145e-5
#define KAPPA 1.1
#define MAX_N 8640
#define MAX_NN 24
#define RDF_BINS 350
#define RDF_MAX 3.5
#define CKPT_MAGIC 0x474D4331u
#define CKPT_VERSION 1

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
static double TER_BETA_N = 0.0;
static double TER_INV_2N = 0.0;

static volatile sig_atomic_t g_stop = 0;

static void on_stop(int sig) {
    (void)sig;
    g_stop = 1;
}

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

static double wrap(double v, double L) {
    v -= L * floor(v / L);
    return v;
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
            if (m >= MAX_NN) break;
            nb[m++] = j;
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
    double b = pow(1.0 + TER_BETA_N * zn, TER_INV_2N);
    double fR = TER_A * exp(-TER_L1 * rij);
    double fA = -TER_B * exp(-TER_L2 * rij);
    return fc * (fR + b * fA);
}

/* Tablas en RAM para no llamar a libm en cada enlace. ~40 MiB, residentes. */
#define LUT_R 65536
#define LUT_G 16384
#define LUT_B_N (1 << 22)
static double lut_fc[LUT_R], lut_fR[LUT_R], lut_fA[LUT_R], lut_g[LUT_G];
static double *lut_b = NULL;
static double lut_r_dr, lut_b_z0, lut_b_dz;
static int lut_ready = 0;

static void lut_init(void) {
    if (lut_ready) return;
    lut_r_dr = TER_S / (double)(LUT_R - 1);
    for (int i = 0; i < LUT_R; i++) {
        double r = i * lut_r_dr;
        double fc;
        if (r < TER_R) fc = 1.0;
        else if (r >= TER_S) fc = 0.0;
        else {
            double x = M_PI * (r - TER_R) / (TER_S - TER_R);
            fc = 0.5 + 0.5 * cos(x);
        }
        lut_fc[i] = fc;
        lut_fR[i] = TER_A * exp(-TER_L1 * r);
        lut_fA[i] = -TER_B * exp(-TER_L2 * r);
    }
    for (int i = 0; i < LUT_G; i++) {
        double cost = -1.0 + 2.0 * (double)i / (double)(LUT_G - 1);
        lut_g[i] = g_theta(cost);
    }
    lut_b = (double *)malloc((size_t)LUT_B_N * sizeof(double));
    if (!lut_b) return;
    lut_b_z0 = -12.0;
    double z1 = 24.0;
    lut_b_dz = (z1 - lut_b_z0) / (double)(LUT_B_N - 1);
    for (int i = 0; i < LUT_B_N; i++) {
        double lz = lut_b_z0 + i * lut_b_dz;
        double zn = exp(TER_N * lz);
        lut_b[i] = exp(TER_INV_2N * log(1.0 + TER_BETA_N * zn));
    }
    /* Fija las páginas: la tabla tiene que quedarse en RAM durante la corrida. */
    volatile double sink = 0.0;
    for (int i = 0; i < LUT_B_N; i += 512) sink += lut_b[i];
    if (sink == 1e300) fprintf(stderr, "lut\n");
    lut_ready = 1;
}

static inline double lut_interp(const double *tab, int n, double x) {
    if (x <= 0.0) return tab[0];
    if (x >= (double)(n - 1)) return tab[n - 1];
    int i = (int)x;
    double t = x - (double)i;
    return tab[i] * (1.0 - t) + tab[i + 1] * t;
}

static inline void radial_fast(double r, double *fc, double *fR, double *fA) {
    double x = r / lut_r_dr;
    *fc = lut_interp(lut_fc, LUT_R, x);
    *fR = lut_interp(lut_fR, LUT_R, x);
    *fA = lut_interp(lut_fA, LUT_R, x);
}

static inline double g_fast(double cost) {
    if (cost > 1.0) cost = 1.0;
    if (cost < -1.0) cost = -1.0;
    double x = (cost + 1.0) * 0.5 * (double)(LUT_G - 1);
    return lut_interp(lut_g, LUT_G, x);
}

static inline double b_fast(double zeta) {
    if (!(zeta > 0.0) || !lut_b) return 1.0;
    double lz = log(zeta);
    double x = (lz - lut_b_z0) / lut_b_dz;
    return lut_interp(lut_b, LUT_B_N, x);
}

static double pair_term_fast(const double *x, const double *y, const double *z,
                             int i, int j, const int *nb, int nn, double lx, double ly) {
    double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
    min_image(&dx, &dy, lx, ly);
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= TER_S * TER_S || r2 < 1e-24) return 0.0;
    double rij = sqrt(r2);
    double fc, fR, fA;
    radial_fast(rij, &fc, &fR, &fA);
    if (fc == 0.0) return 0.0;
    double zeta = 0.0;
    for (int a = 0; a < nn; a++) {
        int k = nb[a];
        if (k == j) continue;
        double dxk = x[k] - x[i], dyk = y[k] - y[i], dzk = z[k] - z[i];
        min_image(&dxk, &dyk, lx, ly);
        double rk2 = dxk * dxk + dyk * dyk + dzk * dzk;
        if (rk2 >= TER_S * TER_S || rk2 < 1e-24) continue;
        double rik = sqrt(rk2);
        double fck, fkR, fkA;
        radial_fast(rik, &fck, &fkR, &fkA);
        if (fck == 0.0) continue;
        double cost = (dx * dxk + dy * dyk + dz * dzk) / (rij * rik);
        zeta += fck * g_fast(cost);
    }
    return fc * (fR + b_fast(zeta) * fA);
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

static double local_energy_brute(const double *x, const double *y, const double *z,
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

typedef struct {
    int nx, ny;
} Mode;

typedef struct {
    int n, p, q;
    double lx, ly;
    double *x, *y, *z;
    int *nn;
    int *neigh;
    int ncx, ncy;
    double cell_w, cell_h;
    int *head;
    int *link;
    int *stamp;
    int stamp_id;
    int cells_cap;
    int overflow;
    Mode *modes;
    int nmodes;
    double *gq_sum;
    double *q_sum;
    int *gq_count;
    double rdf[RDF_BINS];
} Workspace;

static int *nb_ptr(Workspace *w, int i) { return w->neigh + i * MAX_NN; }

static double dist2_ij(const Workspace *w, int i, int j) {
    double dx = w->x[j] - w->x[i];
    double dy = w->y[j] - w->y[i];
    double dz = w->z[j] - w->z[i];
    min_image(&dx, &dy, w->lx, w->ly);
    return dx * dx + dy * dy + dz * dz;
}

static int cell_of(const Workspace *w, int i) {
    double x = wrap(w->x[i], w->lx);
    double y = wrap(w->y[i], w->ly);
    int ix = (int)(x / w->cell_w);
    int iy = (int)(y / w->cell_h);
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (ix >= w->ncx) ix = w->ncx - 1;
    if (iy >= w->ncy) iy = w->ncy - 1;
    return iy * w->ncx + ix;
}

static void cells_bind(Workspace *w) {
    int ncx = (int)floor(w->lx / TER_S);
    int ncy = (int)floor(w->ly / TER_S);
    if (ncx < 1) ncx = 1;
    if (ncy < 1) ncy = 1;
    int need = ncx * ncy;
    if (need > w->cells_cap) {
        free(w->head);
        free(w->stamp);
        w->head = (int *)malloc((size_t)need * sizeof(int));
        w->stamp = (int *)calloc((size_t)need, sizeof(int));
        w->cells_cap = need;
        if (!w->head || !w->stamp) {
            w->overflow = 3;
            return;
        }
    }
    memset(w->stamp, 0, (size_t)w->cells_cap * sizeof(int));
    w->stamp_id = 0;
    w->ncx = ncx;
    w->ncy = ncy;
    w->cell_w = w->lx / ncx;
    w->cell_h = w->ly / ncy;
    for (int k = 0; k < need; k++) w->head[k] = -1;
    for (int i = 0; i < w->n; i++) {
        int c = cell_of(w, i);
        w->link[i] = w->head[c];
        w->head[c] = i;
    }
}

static int begin_stamp(Workspace *w) {
    w->stamp_id++;
    if (w->stamp_id <= 0) {
        memset(w->stamp, 0, (size_t)w->cells_cap * sizeof(int));
        w->stamp_id = 1;
    }
    return w->stamp_id;
}

static void neigh_remove(Workspace *w, int host, int other) {
    int m = w->nn[host];
    int *nb = nb_ptr(w, host);
    for (int a = 0; a < m; a++) {
        if (nb[a] == other) {
            nb[a] = nb[m - 1];
            w->nn[host] = m - 1;
            return;
        }
    }
}

static void neigh_add(Workspace *w, int host, int other) {
    int *nb = nb_ptr(w, host);
    for (int a = 0; a < w->nn[host]; a++) {
        if (nb[a] == other) return;
    }
    if (w->nn[host] >= MAX_NN) {
        w->overflow = 1;
        return;
    }
    nb[w->nn[host]++] = other;
}

static void for_each_near(Workspace *w, int i, int radx, int rady,
                          void (*fn)(Workspace *, int, int, void *), void *ud) {
    int cx, cy;
    int c0 = cell_of(w, i);
    cx = c0 % w->ncx;
    cy = c0 / w->ncx;
    int sid = begin_stamp(w);
    for (int dy = -rady; dy <= rady; dy++) {
        int iy = cy + dy;
        iy %= w->ncy;
        if (iy < 0) iy += w->ncy;
        for (int dx = -radx; dx <= radx; dx++) {
            int ix = cx + dx;
            ix %= w->ncx;
            if (ix < 0) ix += w->ncx;
            int k = iy * w->ncx + ix;
            if (w->stamp[k] == sid) continue;
            w->stamp[k] = sid;
            for (int j = w->head[k]; j != -1; j = w->link[j]) {
                if (j == i) continue;
                fn(w, i, j, ud);
            }
        }
    }
}

static void add_pair_if_close(Workspace *w, int i, int j, void *ud) {
    (void)ud;
    if (j <= i) return;
    if (dist2_ij(w, i, j) < TER_S * TER_S) {
        neigh_add(w, i, j);
        neigh_add(w, j, i);
    }
}

static void neighbors_all(Workspace *w) {
    memset(w->nn, 0, (size_t)w->n * sizeof(int));
    int radx = 1, rady = 1;
    if (w->cell_w > 0.0 && TER_S > w->cell_w)
        radx = (int)ceil(TER_S / w->cell_w);
    if (w->cell_h > 0.0 && TER_S > w->cell_h)
        rady = (int)ceil(TER_S / w->cell_h);
    for (int i = 0; i < w->n; i++)
        for_each_near(w, i, radx, rady, add_pair_if_close, NULL);
}

static void relink_neighbor(Workspace *w, int i, int j, void *ud) {
    (void)ud;
    if (dist2_ij(w, i, j) < TER_S * TER_S) {
        neigh_add(w, i, j);
        neigh_add(w, j, i);
    }
}

static void rebuild_atom(Workspace *w, int i) {
    int oldn = w->nn[i];
    int old[MAX_NN];
    memcpy(old, nb_ptr(w, i), (size_t)oldn * sizeof(int));
    for (int a = 0; a < oldn; a++) neigh_remove(w, old[a], i);
    w->nn[i] = 0;
    int radx = 1, rady = 1;
    if (TER_S > w->cell_w) radx = (int)ceil(TER_S / w->cell_w);
    if (TER_S > w->cell_h) rady = (int)ceil(TER_S / w->cell_h);
    for_each_near(w, i, radx, rady, relink_neighbor, NULL);
}

/* Igual que rebuild_atom, sin el sello global: lo pueden correr varios hilos
 * si los átomos están a más de dos veces el alcance. */
static void rebuild_atom_safe(Workspace *w, int i) {
    int oldn = w->nn[i];
    int old[MAX_NN];
    memcpy(old, nb_ptr(w, i), (size_t)oldn * sizeof(int));
    for (int a = 0; a < oldn; a++) neigh_remove(w, old[a], i);
    w->nn[i] = 0;
    int radx = 1, rady = 1;
    if (TER_S > w->cell_w) radx = (int)ceil(TER_S / w->cell_w);
    if (TER_S > w->cell_h) rady = (int)ceil(TER_S / w->cell_h);
    int c0 = cell_of(w, i);
    int cx = c0 % w->ncx;
    int cy = c0 / w->ncx;
    int visited[64];
    int nv = 0;
    const double s2 = TER_S * TER_S;
    for (int dy = -rady; dy <= rady; dy++) {
        int iy = cy + dy;
        iy %= w->ncy;
        if (iy < 0) iy += w->ncy;
        for (int dx = -radx; dx <= radx; dx++) {
            int ix = cx + dx;
            ix %= w->ncx;
            if (ix < 0) ix += w->ncx;
            int k = iy * w->ncx + ix;
            int seen = 0;
            for (int t = 0; t < nv; t++)
                if (visited[t] == k) seen = 1;
            if (seen) continue;
            if (nv < 64) visited[nv++] = k;
            for (int j = w->head[k]; j != -1; j = w->link[j]) {
                if (j == i) continue;
                if (dist2_ij(w, i, j) < s2) {
                    neigh_add(w, i, j);
                    neigh_add(w, j, i);
                }
            }
        }
    }
}

static void cell_remove(Workspace *w, int i, int c_old) {
    int *pp = &w->head[c_old];
    while (*pp != -1 && *pp != i) pp = &w->link[*pp];
    if (*pp != i) {
        w->overflow = 2;
        return;
    }
    *pp = w->link[i];
}

static void move_atom_cell(Workspace *w, int i, int c_old) {
    int c_new = cell_of(w, i);
    if (c_old == c_new) return;
    cell_remove(w, i, c_old);
    w->link[i] = w->head[c_new];
    w->head[c_new] = i;
}

static double energy_ws(const Workspace *w) {
    double e = 0.0;
    for (int i = 0; i < w->n; i++) {
        int ni = w->nn[i];
        const int *nb = w->neigh + i * MAX_NN;
        for (int a = 0; a < ni; a++)
            e += 0.5 * pair_term_nb(w->x, w->y, w->z, i, nb[a], nb, ni, w->lx, w->ly);
    }
    return e;
}

static double local_energy_ws(const Workspace *w, int i) {
    double e = 0.0;
    int ni = w->nn[i];
    const int *nbi = w->neigh + i * MAX_NN;
    for (int a = 0; a < ni; a++) {
        int j = nbi[a];
        e += 0.5 * pair_term_nb(w->x, w->y, w->z, i, j, nbi, ni, w->lx, w->ly);
        int nj = w->nn[j];
        const int *nbj = w->neigh + j * MAX_NN;
        e += 0.5 * pair_term_nb(w->x, w->y, w->z, j, i, nbj, nj, w->lx, w->ly);
        for (int b = 0; b < nj; b++) {
            int k = nbj[b];
            if (k == i) continue;
            e += 0.5 * pair_term_nb(w->x, w->y, w->z, j, k, nbj, nj, w->lx, w->ly);
        }
    }
    return e;
}

static double local_energy_fast(const Workspace *w, int i) {
    double e = 0.0;
    int ni = w->nn[i];
    const int *nbi = w->neigh + i * MAX_NN;
    for (int a = 0; a < ni; a++) {
        int j = nbi[a];
        e += 0.5 * pair_term_fast(w->x, w->y, w->z, i, j, nbi, ni, w->lx, w->ly);
        int nj = w->nn[j];
        const int *nbj = w->neigh + j * MAX_NN;
        e += 0.5 * pair_term_fast(w->x, w->y, w->z, j, i, nbj, nj, w->lx, w->ly);
        for (int b = 0; b < nj; b++) {
            int k = nbj[b];
            if (k == i) continue;
            e += 0.5 * pair_term_fast(w->x, w->y, w->z, j, k, nbj, nj, w->lx, w->ly);
        }
    }
    return e;
}

static double energy_fast(Workspace *w) {
    if (!lut_ready) lut_init();
    double e = 0.0;
    int n = w->n;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : e) schedule(static) if (n > 256)
#endif
    for (int i = 0; i < n; i++) {
        int ni = w->nn[i];
        const int *nb = w->neigh + i * MAX_NN;
        double ei = 0.0;
        for (int a = 0; a < ni; a++)
            ei += 0.5 * pair_term_fast(w->x, w->y, w->z, i, nb[a], nb, ni, w->lx, w->ly);
        e += ei;
    }
    return e;
}

static int state_ok(const Workspace *w, double energy) {
    if (!isfinite(energy) || !isfinite(w->lx) || !isfinite(w->ly)) return 0;
    if (w->lx < 1.0 || w->ly < 1.0) return 0;
    if (w->overflow) return 0;
    double zmax = 0.0;
    for (int i = 0; i < w->n; i++) {
        if (!isfinite(w->x[i]) || !isfinite(w->y[i]) || !isfinite(w->z[i])) return 0;
        double az = fabs(w->z[i]);
        if (az > zmax) zmax = az;
    }
    if (zmax > 25.0) return 0;
    return 1;
}

static int want_mode(int nx, int ny, double q) {
    if (nx == 0 && ny == 0) return 0;
    /* Zona del máximo de 70 Å y de la ley armónica 1/q². */
    if (q <= 1.25) return 1;
    /* Picos de Bragg del paper: 4π/(3a) ≈ 2.95 y 4π/(√3 a) ≈ 5.11. */
    if (fabs(q - 2.949385) < 0.12) return 1;
    if (fabs(q - 5.108225) < 0.15) return 1;
    if (q <= 6.05) {
        unsigned h = (unsigned)(nx * 73856093) ^ (unsigned)(ny * 19349663);
        if ((h % 32u) == 0u) return 1;
    }
    return 0;
}

static int build_modes(Workspace *w) {
    int nx_lim = (int)(6.2 * w->lx / (2.0 * M_PI)) + 2;
    int ny_lim = (int)(6.2 * w->ly / (2.0 * M_PI)) + 2;
    if (nx_lim < 2) nx_lim = 2;
    if (ny_lim < 2) ny_lim = 2;
    int cap = 0;
    for (int nx = -nx_lim; nx <= nx_lim; nx++) {
        for (int ny = -ny_lim; ny <= ny_lim; ny++) {
            double qx = 2.0 * M_PI * nx / w->lx;
            double qy = 2.0 * M_PI * ny / w->ly;
            if (want_mode(nx, ny, hypot(qx, qy))) cap++;
        }
    }
    if (cap < 1) return -1;
    w->modes = (Mode *)malloc((size_t)cap * sizeof(Mode));
    w->gq_sum = (double *)calloc((size_t)cap, sizeof(double));
    w->q_sum = (double *)calloc((size_t)cap, sizeof(double));
    w->gq_count = (int *)calloc((size_t)cap, sizeof(int));
    if (!w->modes || !w->gq_sum || !w->q_sum || !w->gq_count) return -1;
    int m = 0;
    for (int nx = -nx_lim; nx <= nx_lim; nx++) {
        for (int ny = -ny_lim; ny <= ny_lim; ny++) {
            double qx = 2.0 * M_PI * nx / w->lx;
            double qy = 2.0 * M_PI * ny / w->ly;
            if (!want_mode(nx, ny, hypot(qx, qy))) continue;
            w->modes[m].nx = nx;
            w->modes[m].ny = ny;
            m++;
        }
    }
    w->nmodes = m;
    return 0;
}

static Workspace *ws_alloc(int n) {
    Workspace *w = (Workspace *)calloc(1, sizeof(Workspace));
    if (!w) return NULL;
    w->n = n;
    w->x = (double *)calloc((size_t)n, sizeof(double));
    w->y = (double *)calloc((size_t)n, sizeof(double));
    w->z = (double *)calloc((size_t)n, sizeof(double));
    w->nn = (int *)calloc((size_t)n, sizeof(int));
    w->neigh = (int *)calloc((size_t)n * MAX_NN, sizeof(int));
    w->link = (int *)malloc((size_t)n * sizeof(int));
    if (!w->x || !w->y || !w->z || !w->nn || !w->neigh || !w->link) {
        free(w->x); free(w->y); free(w->z); free(w->nn); free(w->neigh); free(w->link); free(w);
        return NULL;
    }
    return w;
}

static void ws_free(Workspace *w) {
    if (!w) return;
    free(w->x); free(w->y); free(w->z);
    free(w->nn); free(w->neigh); free(w->link);
    free(w->head); free(w->stamp);
    free(w->modes); free(w->gq_sum); free(w->q_sum); free(w->gq_count);
    free(w);
}

static int ws_bind_geometry(Workspace *w) {
    cells_bind(w);
    if (w->overflow) return -1;
    neighbors_all(w);
    if (w->overflow) return -1;
    return 0;
}

typedef struct {
    const char *path;
    int replica;
    int step;
    int total;
    double energy;
    double acc;
    double h;
} Beat;

static void beat_write(const Beat *b, const char *phase) {
    if (!b || !b->path || !b->path[0]) return;
    char tmp[768];
    snprintf(tmp, sizeof(tmp), "%s.tmp", b->path);
    FILE *f = fopen(tmp, "w");
    if (!f) return;
    fprintf(f,
            "replica=%d\nstep=%d\ntotal=%d\nenergy=%.8f\nacc=%.6f\nh=%.6f\n"
            "phase=%s\nunix=%ld\npid=%d\n",
            b->replica, b->step, b->total, b->energy, b->acc, b->h, phase,
            (long)time(NULL), (int)getpid());
    fflush(f);
    fclose(f);
    rename(tmp, b->path);
}

static int ckpt_write(const char *path, const Workspace *w, int replica, int step,
                      int n_eq, int n_prod, int sample_every, double T, double energy,
                      double step_amp, double area_step, double h_rms_acc, double h_rms_eq,
                      int64_t acc_m, int64_t try_m, int64_t acc_a, int64_t try_a,
                      int nsamp, uint64_t rng) {
    if (!path || !path[0]) return 0;
    char tmp[768], prev[768];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    snprintf(prev, sizeof(prev), "%s.prev", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    uint32_t magic = CKPT_MAGIC, version = CKPT_VERSION;
    int32_t header_i[10];
    header_i[0] = replica;
    header_i[1] = step;
    header_i[2] = w->n;
    header_i[3] = w->p;
    header_i[4] = w->q;
    header_i[5] = n_eq;
    header_i[6] = n_prod;
    header_i[7] = sample_every;
    header_i[8] = w->nmodes;
    header_i[9] = nsamp;
    double header_d[8] = {T, w->lx, w->ly, energy, step_amp, area_step, h_rms_acc, h_rms_eq};
    int64_t counts[4] = {acc_m, try_m, acc_a, try_a};
    int ok = 1;
    ok = ok && fwrite(&magic, 4, 1, f) == 1;
    ok = ok && fwrite(&version, 4, 1, f) == 1;
    ok = ok && fwrite(header_i, sizeof(header_i), 1, f) == 1;
    ok = ok && fwrite(header_d, sizeof(header_d), 1, f) == 1;
    ok = ok && fwrite(counts, sizeof(counts), 1, f) == 1;
    ok = ok && fwrite(&rng, sizeof(rng), 1, f) == 1;
    ok = ok && fwrite(w->x, sizeof(double), (size_t)w->n, f) == (size_t)w->n;
    ok = ok && fwrite(w->y, sizeof(double), (size_t)w->n, f) == (size_t)w->n;
    ok = ok && fwrite(w->z, sizeof(double), (size_t)w->n, f) == (size_t)w->n;
    for (int m = 0; ok && m < w->nmodes; m++) {
        int32_t nx = w->modes[m].nx, ny = w->modes[m].ny;
        ok = ok && fwrite(&nx, 4, 1, f) == 1 && fwrite(&ny, 4, 1, f) == 1;
    }
    ok = ok && fwrite(w->gq_sum, sizeof(double), (size_t)w->nmodes, f) == (size_t)w->nmodes;
    ok = ok && fwrite(w->q_sum, sizeof(double), (size_t)w->nmodes, f) == (size_t)w->nmodes;
    ok = ok && fwrite(w->gq_count, sizeof(int), (size_t)w->nmodes, f) == (size_t)w->nmodes;
    ok = ok && fwrite(w->rdf, sizeof(double), RDF_BINS, f) == RDF_BINS;
    if (ok) fflush(f);
    if (ok && fsync(fileno(f)) != 0) ok = 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        unlink(tmp);
        return -1;
    }
    if (access(path, F_OK) == 0) rename(path, prev);
    if (rename(tmp, path) != 0) return -1;
    return 0;
}

static int ckpt_read(const char *path, Workspace *w, int replica, int n_eq, int n_prod,
                     int sample_every, double T, int *step, double *energy, double *step_amp,
                     double *area_step, double *h_rms_acc, double *h_rms_eq, int64_t *acc_m,
                     int64_t *try_m, int64_t *acc_a, int64_t *try_a, int *nsamp, uint64_t *rng) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic = 0, version = 0;
    int32_t header_i[10];
    double header_d[8];
    int64_t counts[4];
    int ok = 1;
    ok = ok && fread(&magic, 4, 1, f) == 1 && magic == CKPT_MAGIC;
    ok = ok && fread(&version, 4, 1, f) == 1 && version == CKPT_VERSION;
    ok = ok && fread(header_i, sizeof(header_i), 1, f) == 1;
    ok = ok && fread(header_d, sizeof(header_d), 1, f) == 1;
    ok = ok && fread(counts, sizeof(counts), 1, f) == 1;
    ok = ok && fread(rng, sizeof(*rng), 1, f) == 1;
    ok = ok && header_i[0] == replica && header_i[2] == w->n;
    ok = ok && header_i[5] == n_eq && header_i[6] == n_prod && header_i[7] == sample_every;
    ok = ok && header_i[8] == w->nmodes;
    ok = ok && fabs(header_d[0] - T) < 1e-9;
    if (ok) {
        ok = ok && fread(w->x, sizeof(double), (size_t)w->n, f) == (size_t)w->n;
        ok = ok && fread(w->y, sizeof(double), (size_t)w->n, f) == (size_t)w->n;
        ok = ok && fread(w->z, sizeof(double), (size_t)w->n, f) == (size_t)w->n;
    }
    for (int m = 0; ok && m < w->nmodes; m++) {
        int32_t nx = 0, ny = 0;
        ok = ok && fread(&nx, 4, 1, f) == 1 && fread(&ny, 4, 1, f) == 1;
        ok = ok && nx == w->modes[m].nx && ny == w->modes[m].ny;
    }
    if (ok) {
        ok = ok && fread(w->gq_sum, sizeof(double), (size_t)w->nmodes, f) == (size_t)w->nmodes;
        ok = ok && fread(w->q_sum, sizeof(double), (size_t)w->nmodes, f) == (size_t)w->nmodes;
        ok = ok && fread(w->gq_count, sizeof(int), (size_t)w->nmodes, f) == (size_t)w->nmodes;
        ok = ok && fread(w->rdf, sizeof(double), RDF_BINS, f) == RDF_BINS;
    }
    fclose(f);
    if (!ok) return -1;
    w->lx = header_d[1];
    w->ly = header_d[2];
    *energy = header_d[3];
    *step_amp = header_d[4];
    *area_step = header_d[5];
    *h_rms_acc = header_d[6];
    *h_rms_eq = header_d[7];
    *acc_m = counts[0];
    *try_m = counts[1];
    *acc_a = counts[2];
    *try_a = counts[3];
    *nsamp = header_i[9];
    *step = header_i[1];
    return 0;
}

static void sample_observables(Workspace *w, Beat *beat) {
    (void)beat;
    int n = w->n;
    double zmean = 0.0;
    for (int i = 0; i < n; i++) zmean += w->z[i];
    zmean /= n;
    double h2 = 0.0;
    double *h = (double *)malloc((size_t)n * sizeof(double));
    if (!h) return;
    for (int i = 0; i < n; i++) {
        h[i] = w->z[i] - zmean;
        h2 += h[i] * h[i];
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (w->nmodes > 32)
#endif
    for (int m = 0; m < w->nmodes; m++) {
        double qx = 2.0 * M_PI * w->modes[m].nx / w->lx;
        double qy = 2.0 * M_PI * w->modes[m].ny / w->ly;
        double q = hypot(qx, qy);
        double hr = 0.0, hi = 0.0;
        for (int j = 0; j < n; j++) {
            double ph = qx * w->x[j] + qy * w->y[j];
            hr += h[j] * cos(ph);
            hi -= h[j] * sin(ph);
        }
        double g = q * q * (hr * hr + hi * hi);
        w->gq_sum[m] += g / n;
        w->q_sum[m] += q;
        w->gq_count[m] += 1;
    }
    free(h);
    int radx = (int)ceil(RDF_MAX / w->cell_w);
    int rady = (int)ceil(RDF_MAX / w->cell_h);
    if (radx < 1) radx = 1;
    if (rady < 1) rady = 1;
    double dr = RDF_MAX / RDF_BINS;
    const double rmax2 = RDF_MAX * RDF_MAX;
    for (int i = 0; i < n; i++) {
        int cx = cell_of(w, i) % w->ncx;
        int cy = cell_of(w, i) / w->ncx;
        int sid = begin_stamp(w);
        for (int dy = -rady; dy <= rady; dy++) {
            int iy = cy + dy;
            iy %= w->ncy;
            if (iy < 0) iy += w->ncy;
            for (int dx = -radx; dx <= radx; dx++) {
                int ix = cx + dx;
                ix %= w->ncx;
                if (ix < 0) ix += w->ncx;
                int k = iy * w->ncx + ix;
                if (w->stamp[k] == sid) continue;
                w->stamp[k] = sid;
                for (int j = w->head[k]; j != -1; j = w->link[j]) {
                    if (j <= i) continue;
                    double r2 = dist2_ij(w, i, j);
                    if (r2 >= rmax2) continue;
                    int b = (int)(sqrt(r2) / dr);
                    if (b >= 0 && b < RDF_BINS) w->rdf[b] += 2.0;
                }
            }
        }
    }
    (void)h2;
}

static double height_rms(const Workspace *w) {
    double zmean = 0.0;
    for (int i = 0; i < w->n; i++) zmean += w->z[i];
    zmean /= w->n;
    double h2 = 0.0;
    for (int i = 0; i < w->n; i++) {
        double hi = w->z[i] - zmean;
        h2 += hi * hi;
    }
    return sqrt(h2 / w->n);
}

typedef struct {
    int replica, p, q, n_eq, n_prod, sample_every, ckpt_every, beat_every, threads;
    double T;
    uint64_t seed;
    const char *outdir;
    const char *heartbeat;
    const char *checkpoint;
    const char *resume;
} RunCfg;

static int write_outputs(const Workspace *w, const RunCfg *cfg, double energy, double h_rms,
                         double h_rms_eq, double step_amp, int64_t acc_m, int64_t try_m,
                         int64_t acc_a, int64_t try_a, int nsamp) {
    char path[768];
    snprintf(path, sizeof(path), "%s/replica_%02d_summary.csv", cfg->outdir, cfg->replica);
    FILE *sum = fopen(path, "w");
    if (!sum) return -1;
    fprintf(sum,
            "replica,N,T,n_eq,n_prod,energy,h_rms,h_rms_eq,lx,ly,S0,"
            "acc_move,acc_area,nsamples,step_amp\n");
    fprintf(sum, "%d,%d,%.2f,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.6f,%d,%.5f\n",
            cfg->replica, w->n, cfg->T, cfg->n_eq, cfg->n_prod, energy, h_rms, h_rms_eq, w->lx,
            w->ly, w->lx * w->ly / w->n, try_m ? (double)acc_m / (double)try_m : 0.0,
            try_a ? (double)acc_a / (double)try_a : 0.0, nsamp, step_amp);
    fclose(sum);

    snprintf(path, sizeof(path), "%s/replica_%02d_gq.csv", cfg->outdir, cfg->replica);
    FILE *gf = fopen(path, "w");
    if (!gf) return -1;
    fprintf(gf, "nx,ny,q,G_over_N\n");
    for (int m = 0; m < w->nmodes; m++) {
        double q = (w->gq_count[m] > 0) ? w->q_sum[m] / w->gq_count[m] : 0.0;
        double g = (w->gq_count[m] > 0) ? w->gq_sum[m] / w->gq_count[m] : 0.0;
        fprintf(gf, "%d,%d,%.8f,%.8e\n", w->modes[m].nx, w->modes[m].ny, q, g);
    }
    fclose(gf);

    snprintf(path, sizeof(path), "%s/replica_%02d_rdf.csv", cfg->outdir, cfg->replica);
    FILE *rf = fopen(path, "w");
    if (!rf) return -1;
    fprintf(rf, "r,g\n");
    double density = w->n / (w->lx * w->ly);
    double dr = RDF_MAX / RDF_BINS;
    for (int b = 0; b < RDF_BINS; b++) {
        double r = (b + 0.5) * dr;
        double shell = 2.0 * M_PI * r * dr * density;
        double g = (nsamp > 0 && shell > 0.0) ? w->rdf[b] / (nsamp * (double)w->n * shell) : 0.0;
        fprintf(rf, "%.6f,%.8f\n", r, g);
    }
    fclose(rf);

    snprintf(path, sizeof(path), "%s/replica_%02d_snapshot.csv", cfg->outdir, cfg->replica);
    FILE *sf = fopen(path, "w");
    if (!sf) return -1;
    fprintf(sf, "x,y,z,lx,ly\n");
    for (int i = 0; i < w->n; i++)
        fprintf(sf, "%.8f,%.8f,%.8f,%.8f,%.8f\n", w->x[i], w->y[i], w->z[i], w->lx, w->ly);
    fclose(sf);

    snprintf(path, sizeof(path), "%s/replica_%02d.done", cfg->outdir, cfg->replica);
    FILE *df = fopen(path, "w");
    if (!df) return -1;
    fprintf(df, "ok replica=%d N=%d nsamples=%d\n", cfg->replica, w->n, nsamp);
    fflush(df);
    fsync(fileno(df));
    fclose(df);
    return 0;
}

#define SEP_MIN 9.5
#define SEP_MIN2 (SEP_MIN * SEP_MIN)

typedef struct {
    int i;
    double dx, dy, dz, u, dE;
    int accept;
} Prop;

static int separated(const Workspace *w, int i, const Prop *p, int nb) {
    for (int b = 0; b < nb; b++) {
        if (dist2_ij(w, i, p[b].i) < SEP_MIN2) return 0;
    }
    return 1;
}

static void apply_prop(Workspace *w, Prop *pr, double beta) {
    int i = pr->i;
    double e_old = local_energy_fast(w, i);
    double ox = w->x[i], oy = w->y[i], oz = w->z[i];
    int c_old = cell_of(w, i);
    w->x[i] = wrap(ox + pr->dx, w->lx);
    w->y[i] = wrap(oy + pr->dy, w->ly);
    w->z[i] = oz + pr->dz;
    move_atom_cell(w, i, c_old);
    rebuild_atom_safe(w, i);
    double e_new = local_energy_fast(w, i);
    double dE = e_new - e_old;
    pr->dE = dE;
    pr->accept = isfinite(dE) && (dE <= 0.0 || pr->u < exp(-beta * dE));
    if (!pr->accept) {
        int c_bad = cell_of(w, i);
        w->x[i] = ox;
        w->y[i] = oy;
        w->z[i] = oz;
        move_atom_cell(w, i, c_bad);
        rebuild_atom_safe(w, i);
    }
}

/* Un paso MC = N intentos. Los que quedan a más de SEP_MIN se evalúan a la vez. */
static void mc_moves(Workspace *w, Rng *rng, int n_attempts, double step, double beta,
                     double *energy, int64_t *acc_move, int64_t *try_move, int threads) {
    int n = w->n;
    int width = threads > 1 ? threads : 1;
    if (width > 128) width = 128;
    if (width > n) width = n;
    Prop *props = (Prop *)malloc((size_t)width * sizeof(Prop));
    if (!props) return;
    int remain = n_attempts;
    while (remain > 0 && !w->overflow) {
        int limit = width < remain ? width : remain;
        int nb = 0;
        int tries = 0;
        while (nb < limit && tries < n * 4) {
            tries++;
            int i = (int)(rng_uni(rng) * n);
            if (i >= n) i = n - 1;
            if (!separated(w, i, props, nb)) continue;
            props[nb].i = i;
            props[nb].dx = step * rng_sym(rng);
            props[nb].dy = step * rng_sym(rng);
            props[nb].dz = step * rng_sym(rng);
            props[nb].u = rng_uni(rng);
            props[nb].accept = 0;
            nb++;
        }
        if (nb == 0) break;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (threads > 1 && nb > 1) num_threads(threads)
#endif
        for (int b = 0; b < nb; b++) apply_prop(w, &props[b], beta);
        for (int b = 0; b < nb; b++) {
            (*try_move)++;
            if (props[b].accept) {
                *energy += props[b].dE;
                (*acc_move)++;
            }
        }
        remain -= nb;
    }
    free(props);
}

static int run_replica(const RunCfg *cfg) {
    int n_expect = 4 * cfg->p * cfg->q;
    if (n_expect <= 0 || n_expect > MAX_N) {
        fprintf(stderr, "N=%d fuera de 1..%d\n", n_expect, MAX_N);
        return 1;
    }
    Workspace *w = ws_alloc(n_expect);
    if (!w) return 1;
    w->p = cfg->p;
    w->q = cfg->q;
    make_lattice(w->x, w->y, w->z, cfg->p, cfg->q, &w->lx, &w->ly, &w->n);
    if (build_modes(w) != 0) {
        ws_free(w);
        return 1;
    }
    if (ws_bind_geometry(w) != 0) {
        fprintf(stderr, "réplica %d: no se pudo construir la lista de vecinos\n", cfg->replica);
        ws_free(w);
        return 1;
    }

    if (!lut_ready) lut_init();
#ifdef _OPENMP
    if (cfg->threads > 1) omp_set_num_threads(cfg->threads);
    else omp_set_num_threads(1);
#endif
    double energy = energy_fast(w);
    double step = 0.04;
    double area_step = 0.002;
    double h_rms_acc = 0.0, h_rms_eq = 0.0;
    int64_t acc_move = 0, try_move = 0, acc_area = 0, try_area = 0;
    int nsamp = 0;
    int start_step = 0;
    Rng rng;
    rng_seed(&rng, cfg->seed + (uint64_t)cfg->replica * 0xD1B54A32D192ED03ULL);

    if (cfg->resume && cfg->resume[0] && access(cfg->resume, R_OK) == 0) {
        int step_done = 0;
        uint64_t rngs = 0;
        if (ckpt_read(cfg->resume, w, cfg->replica, cfg->n_eq, cfg->n_prod, cfg->sample_every,
                      cfg->T, &step_done, &energy, &step, &area_step, &h_rms_acc, &h_rms_eq,
                      &acc_move, &try_move, &acc_area, &try_area, &nsamp, &rngs) != 0) {
            fprintf(stderr, "réplica %d: checkpoint ilegible %s\n", cfg->replica, cfg->resume);
            ws_free(w);
            return 4;
        }
        rng.s = rngs;
        start_step = step_done;
        if (ws_bind_geometry(w) != 0 || !state_ok(w, energy)) {
            fprintf(stderr, "réplica %d: checkpoint con estado inválido\n", cfg->replica);
            ws_free(w);
            return 2;
        }
        fprintf(stdout, "réplica %02d reanuda en el paso %d\n", cfg->replica, start_step);
        fflush(stdout);
    }

    int total = cfg->n_eq + cfg->n_prod;
    int ntrace_cap = total / 200 + 8;
    int *tr_step = (int *)calloc((size_t)ntrace_cap, sizeof(int));
    double *tr_e = (double *)calloc((size_t)ntrace_cap, sizeof(double));
    double *tr_h = (double *)calloc((size_t)ntrace_cap, sizeof(double));
    double *tr_lx = (double *)calloc((size_t)ntrace_cap, sizeof(double));
    double *tr_ly = (double *)calloc((size_t)ntrace_cap, sizeof(double));
    double *tr_accm = (double *)calloc((size_t)ntrace_cap, sizeof(double));
    double *tr_acca = (double *)calloc((size_t)ntrace_cap, sizeof(double));
    int ntrace = 0;
    if (!tr_step || !tr_e || !tr_h || !tr_lx || !tr_ly || !tr_accm || !tr_acca) {
        ws_free(w);
        return 1;
    }

    double beta = 1.0 / (KB_EV * cfg->T);
    char trace_path[768];
    snprintf(trace_path, sizeof(trace_path), "%s/replica_%02d_trace.csv", cfg->outdir, cfg->replica);
    FILE *trace_live = fopen(trace_path, start_step > 0 ? "a" : "w");
    if (!trace_live) {
        ws_free(w);
        return 4;
    }
    if (start_step == 0)
        fprintf(trace_live, "step,energy,h_rms,lx,ly,acc_move,acc_area\n");
    Beat beat = {cfg->heartbeat, cfg->replica, start_step, total, energy, 0.0, 0.0};
    beat_write(&beat, start_step ? "reanuda" : "inicio");
    fprintf(stdout,
            "réplica %02d  N=%d  Lx=%.2f  Ly=%.2f  modos=%d  hilos=%d  pasos=%d..%d\n",
            cfg->replica, w->n, w->lx, w->ly, w->nmodes, cfg->threads > 0 ? cfg->threads : 1,
            start_step + 1, total);
    fflush(stdout);

    double *xs = (double *)malloc((size_t)w->n * sizeof(double));
    double *ys = (double *)malloc((size_t)w->n * sizeof(double));
    if (!xs || !ys) {
        fclose(trace_live);
        free(xs);
        free(ys);
        free(tr_step); free(tr_e); free(tr_h); free(tr_lx); free(tr_ly); free(tr_accm); free(tr_acca);
        ws_free(w);
        return 1;
    }

    int rc = 0;
    for (int istep = start_step + 1; istep <= total; istep++) {
        if (g_stop) {
            beat.step = istep - 1;
            beat.energy = energy;
            beat_write(&beat, "senal");
            if (cfg->checkpoint && cfg->checkpoint[0])
                ckpt_write(cfg->checkpoint, w, cfg->replica, istep - 1, cfg->n_eq, cfg->n_prod,
                           cfg->sample_every, cfg->T, energy, step, area_step, h_rms_acc, h_rms_eq,
                           acc_move, try_move, acc_area, try_area, nsamp, rng.s);
            rc = 75;
            break;
        }
        int n = w->n;
        int threads = cfg->threads > 0 ? cfg->threads : 1;
        mc_moves(w, &rng, n, step, beta, &energy, &acc_move, &try_move, threads);
        if (w->overflow) {
            rc = 3;
            break;
        }

        if (istep % 5 == 0) {
            double factor = exp(area_step * rng_sym(&rng));
            double lx_n = w->lx * factor, ly_n = w->ly * factor;
            double e_old = energy;
            for (int i = 0; i < n; i++) {
                xs[i] = w->x[i];
                ys[i] = w->y[i];
                w->x[i] *= factor;
                w->y[i] *= factor;
            }
            w->lx = lx_n;
            w->ly = ly_n;
            if (ws_bind_geometry(w) != 0) {
                rc = 3;
                break;
            }
            double e_new = energy_fast(w);
            double dE = e_new - e_old;
            double acc_p = exp(-beta * dE + n * 2.0 * log(factor));
            try_area++;
            if (isfinite(e_new) && isfinite(acc_p) && rng_uni(&rng) < acc_p) {
                energy = e_new;
                acc_area++;
            } else {
                for (int i = 0; i < n; i++) {
                    w->x[i] = xs[i];
                    w->y[i] = ys[i];
                }
                w->lx /= factor;
                w->ly /= factor;
                if (ws_bind_geometry(w) != 0) {
                    rc = 3;
                    break;
                }
            }
        }

        if (!state_ok(w, energy)) {
            fprintf(stderr, "réplica %02d: estado no físico en el paso %d (E=%.6g)\n",
                    cfg->replica, istep, energy);
            rc = 2;
            break;
        }

        if (istep == cfg->n_eq) h_rms_eq = height_rms(w);

        if (istep > cfg->n_eq && ((istep - cfg->n_eq) % cfg->sample_every) == 0) {
            beat.step = istep;
            beat.energy = energy;
            beat.acc = try_move ? (double)acc_move / (double)try_move : 0.0;
            beat.h = height_rms(w);
            beat_write(&beat, "muestra");
            sample_observables(w, &beat);
            h_rms_acc += beat.h;
            nsamp++;
        }

        if ((istep % 200 == 0 || istep == total) && ntrace < ntrace_cap) {
            int t = ntrace++;
            tr_step[t] = istep;
            tr_e[t] = energy;
            tr_h[t] = height_rms(w);
            tr_lx[t] = w->lx;
            tr_ly[t] = w->ly;
            tr_accm[t] = try_move ? (double)acc_move / (double)try_move : 0.0;
            tr_acca[t] = try_area ? (double)acc_area / (double)try_area : 0.0;
            fprintf(trace_live, "%d,%.8f,%.8f,%.8f,%.8f,%.6f,%.6f\n", tr_step[t], tr_e[t], tr_h[t],
                    tr_lx[t], tr_ly[t], tr_accm[t], tr_acca[t]);
            fflush(trace_live);
        }

        if (istep % 400 == 0 && try_move > 0) {
            double ar = (double)acc_move / (double)try_move;
            if (ar > 0.55) step *= 1.08;
            if (ar < 0.35) step *= 0.92;
            if (step < 0.005) step = 0.005;
            if (step > 0.12) step = 0.12;
        }

        int do_beat = (cfg->beat_every > 0 && istep % cfg->beat_every == 0) || istep == total;
        if (do_beat) {
            beat.step = istep;
            beat.energy = energy;
            beat.acc = try_move ? (double)acc_move / (double)try_move : 0.0;
            beat.h = (istep <= cfg->n_eq) ? height_rms(w) : (nsamp ? h_rms_acc / nsamp : height_rms(w));
            beat_write(&beat, istep <= cfg->n_eq ? "eq" : "prod");
        }
        if (cfg->ckpt_every > 0 && (istep % cfg->ckpt_every == 0 || istep == cfg->n_eq)) {
            if (ckpt_write(cfg->checkpoint, w, cfg->replica, istep, cfg->n_eq, cfg->n_prod,
                           cfg->sample_every, cfg->T, energy, step, area_step, h_rms_acc, h_rms_eq,
                           acc_move, try_move, acc_area, try_area, nsamp, rng.s) != 0) {
                fprintf(stderr, "réplica %02d: no pude escribir el checkpoint\n", cfg->replica);
                rc = 4;
                break;
            }
        }
    }

    if (rc == 0) {
        double hfin = nsamp ? h_rms_acc / nsamp : 0.0;
        if (write_outputs(w, cfg, energy, hfin, h_rms_eq, step, acc_move, try_move, acc_area,
                          try_area, nsamp) != 0) {
            fprintf(stderr, "réplica %02d: no pude escribir los CSV\n", cfg->replica);
            rc = 4;
        } else {
            printf("replica %02d  E=%.4f eV  h_rms=%.4f Å  acc=%.3f  samples=%d\n",
                   cfg->replica, energy, hfin,
                   try_move ? (double)acc_move / (double)try_move : 0.0, nsamp);
            fflush(stdout);
        }
    }

    fclose(trace_live);
    free(xs);
    free(ys);
    free(tr_step); free(tr_e); free(tr_h); free(tr_lx); free(tr_ly); free(tr_accm); free(tr_acca);
    ws_free(w);
    return rc;
}

static int compare_flat(int p, int q, int expect_n, double lx_ref, double ly_ref, int check_box) {
    int n = 4 * p * q;
    Workspace *w = ws_alloc(n);
    if (!w) return 1;
    w->p = p;
    w->q = q;
    make_lattice(w->x, w->y, w->z, p, q, &w->lx, &w->ly, &w->n);
    if (ws_bind_geometry(w) != 0) {
        ws_free(w);
        return 1;
    }
    int bad_nn = 0;
    for (int i = 0; i < w->n; i++)
        if (w->nn[i] != 3) bad_nn++;
    double e_ws = energy_ws(w);
    double e_br = tersoff_energy(w->x, w->y, w->z, w->n, w->lx, w->ly);
    int i = (n > 7) ? 7 : 0;
    double loc0 = local_energy_ws(w, i);
    double loc0f = local_energy_fast(w, i);
    double loc0b = local_energy_brute(w->x, w->y, w->z, w->n, i, w->lx, w->ly);
    double ox = w->x[i], oy = w->y[i], oz = w->z[i];
    int c0 = cell_of(w, i);
    w->x[i] = wrap(ox + 0.03, w->lx);
    w->y[i] = wrap(oy - 0.02, w->ly);
    w->z[i] = oz + 0.04;
    move_atom_cell(w, i, c0);
    rebuild_atom(w, i);
    double loc1 = local_energy_ws(w, i);
    double loc1f = local_energy_fast(w, i);
    double e_after = energy_ws(w);
    double saved_x = w->x[i], saved_y = w->y[i], saved_z = w->z[i];
    w->x[i] = ox;
    w->y[i] = oy;
    w->z[i] = oz;
    double e_br0 = tersoff_energy(w->x, w->y, w->z, w->n, w->lx, w->ly);
    w->x[i] = saved_x;
    w->y[i] = saved_y;
    w->z[i] = saved_z;
    double e_br1 = tersoff_energy(w->x, w->y, w->z, w->n, w->lx, w->ly);
    double d_full = e_br1 - e_br0;
    double d_loc = loc1 - loc0;
    double d_ws = e_after - e_ws;
    double err = fabs(d_full - d_loc);
    printf("self-test  N=%d  Lx=%.5f  Ly=%.5f  E/N=%.6f eV  nn_malos=%d\n", w->n, w->lx, w->ly,
           e_ws / w->n, bad_nn);
    printf("self-test  dE_full=%.8f  dE_local=%.8f  |diff|=%.3e  |Ews-Ebrute|=%.3e\n", d_full,
           d_loc, err, fabs(e_ws - e_br));
    int pass = 1;
    if (w->n != expect_n) pass = 0;
    if (check_box && fabs(w->lx - lx_ref) > 0.05) pass = 0;
    if (check_box && fabs(w->ly - ly_ref) > 0.05) pass = 0;
    if (e_ws / w->n > -4.0 || e_ws / w->n < -10.0) pass = 0;
    if (bad_nn != 0 || w->overflow) pass = 0;
    if (fabs(e_ws - e_br) > 1e-6) pass = 0;
    if (fabs(loc0 - loc0b) > 1e-6) pass = 0;
    if (err > 1e-6) pass = 0;
    if (fabs(d_ws - d_loc) > 1e-6) pass = 0;
    if (fabs((loc1f - loc0f) - d_loc) > 1e-4) pass = 0;
    if (pass) printf("self-test PASS\n");
    else printf("self-test FAIL\n");
    ws_free(w);
    return pass ? 0 : 1;
}

static int self_test(void) { return compare_flat(10, 6, 240, 24.59, 25.56, 1); }

static int self_test_large(void) {
    /* p=60, q=36 → N=8640, mismo aspecto que N=240, ambos lados > 70 Å. */
    return compare_flat(60, 36, 8640, 0.0, 0.0, 0);
}

int main(int argc, char **argv) {
    TER_BETA_N = pow(TER_BETA, TER_N);
    TER_INV_2N = -0.5 / TER_N;
    lut_init();
    setvbuf(stdout, NULL, _IOLBF, 0);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_stop;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    RunCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.p = 10;
    cfg.q = 6;
    cfg.T = 300.0;
    cfg.n_eq = 400000;
    cfg.n_prod = 500000;
    cfg.sample_every = 0;
    cfg.seed = 42;
    cfg.outdir = "results/datos";
    cfg.replica = -1;
    cfg.ckpt_every = 2000;
    cfg.beat_every = 100;
    cfg.threads = 1;
    int n_replicas = 0;
    int leave_cores = 2;
    int do_self_test = 0;
    int do_self_large = 0;
    int energy_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--T") && i + 1 < argc) cfg.T = atof(argv[++i]);
        else if (!strcmp(argv[i], "--n-eq") && i + 1 < argc) cfg.n_eq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-prod") && i + 1 < argc) cfg.n_prod = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sample-every") && i + 1 < argc)
            cfg.sample_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--replicas") && i + 1 < argc) n_replicas = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            cfg.seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) cfg.outdir = argv[++i];
        else if (!strcmp(argv[i], "--leave-cores") && i + 1 < argc) leave_cores = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--p") && i + 1 < argc) cfg.p = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) cfg.q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--replica") && i + 1 < argc) cfg.replica = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heartbeat") && i + 1 < argc) cfg.heartbeat = argv[++i];
        else if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc) cfg.checkpoint = argv[++i];
        else if (!strcmp(argv[i], "--resume") && i + 1 < argc) cfg.resume = argv[++i];
        else if (!strcmp(argv[i], "--checkpoint-every") && i + 1 < argc)
            cfg.ckpt_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heartbeat-every") && i + 1 < argc)
            cfg.beat_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) cfg.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--self-test")) do_self_test = 1;
        else if (!strcmp(argv[i], "--self-test-large")) do_self_large = 1;
        else if (!strcmp(argv[i], "--energy-only")) energy_only = 1;
        else if (!strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "Usage: graphene_mc [--T 300] [--p 60 --q 36] [--replica K]\n"
                    "       [--n-eq N] [--n-prod N] [--heartbeat FILE] [--checkpoint FILE]\n"
                    "       N = 4*p*q <= 8640. Deja núcleos libres con el lanzador.\n");
            return 0;
        } else {
            fprintf(stderr, "argumento no reconocido: %s\n", argv[i]);
            return 1;
        }
    }

    if (do_self_test) return self_test();
    if (do_self_large) return self_test_large();

    int n_atoms = 4 * cfg.p * cfg.q;
    if (n_atoms <= 0 || n_atoms > MAX_N) {
        fprintf(stderr, "N=%d no permitido (máximo %d)\n", n_atoms, MAX_N);
        return 1;
    }
    if (cfg.T > 300.0 + 1e-9) {
        fprintf(stderr, "T máxima permitida: 300 K\n");
        return 1;
    }
    if (cfg.sample_every <= 0)
        cfg.sample_every = (n_atoms <= 300) ? 10 : (cfg.n_prod > 400 ? cfg.n_prod / 400 : 1);
    if (cfg.sample_every < 1) cfg.sample_every = 1;

    if (energy_only) {
        Workspace *w = ws_alloc(n_atoms);
        if (!w) return 1;
        make_lattice(w->x, w->y, w->z, cfg.p, cfg.q, &w->lx, &w->ly, &w->n);
        if (ws_bind_geometry(w) != 0) return 1;
        double e = energy_ws(w);
        printf("%.12f %.8f %.8f %d\n", e, w->lx, w->ly, w->n);
        ws_free(w);
        return 0;
    }

    if (cfg.replica >= 0) {
        return run_replica(&cfg);
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
    printf("Grafeno MC  N=%d  T=%.1f K  eq=%d  prod=%d  replicas=%d  hilos=%d (de %d; se dejan %d)\n",
           n_atoms, cfg.T, cfg.n_eq, cfg.n_prod, n_replicas, nthreads, nproc, leave_cores);
    fflush(stdout);
    int worst = 0;
#ifdef _OPENMP
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
#endif
    for (int r = 0; r < n_replicas; r++) {
        RunCfg one = cfg;
        one.replica = r;
        one.heartbeat = NULL;
        one.checkpoint = NULL;
        one.resume = NULL;
        int rc = run_replica(&one);
        if (rc > worst) worst = rc;
    }
#ifdef _OPENMP
    printf("tiempo wall = %.1f s\n", omp_get_wtime() - t0);
#endif
    return worst == 0 ? 0 : 1;
}
