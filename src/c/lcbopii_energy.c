/* LCBOPII, PRB 72, 214102 (2005).
 * Corto alcance, conjugación, antienlace, torsión, largo alcance hasta 6 Å
 * y el término de alcance medio, que en la hoja plana de grafito vale cero
 * (Tabla II: la monocapa queda en -7.349 eV/átomo; aquí -7.350). */
#include "lcbopii.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const double A_SR = 53026.92614;
static const double ALPHA = 6.74750993;
static const double B_SR1 = 27618.35706;
static const double BETA1 = 6.34503890;
static const double B_SR2 = 34.07142502;
static const double BETA2 = 1.19712839;
static const double GMIN = 0.0020588719;
static const double GGR = 0.0831047003;
static const double GMAX = 16.0;
static const double G1C[3] = {0.7233666272, 1.7334665088, 1.8701997632};
/* g2 de la Tabla I (0.73994527795, -1.999211817, -17.43251545, -33.96127110,
 * -44.65392079) no es continuo en [-1/2, -1/3]. El puente está en g1(). */
static const double G3C[5] = {-15.19, -25.6168552398, -21.51728397, 0.9899080993, 13.66416160};
static const double AY0 = -0.4, BY0 = 0.01875;
static const double AG = 5.6304664723, BG = 0.1516943990, CG = 0.009832975891;
static const double DG = -0.189175977654, EG = 0.050977653631;
static const double DH = 0.14, C1 = 3.335, C4 = 220.0;
static double C6, R0H, R1H, LH, KAPPA;
static const double F0[4][4] = {
    {0.0, 0.0207, -0.0046, -0.1278},
    {0.0207, 0.0, -0.0365, -0.1043},
    {-0.0046, -0.0365, 0.0, -0.0273},
    {-0.1278, -0.1043, -0.0273, 0.0},
};
static const double F1[4][4] = {
    {0.0, 0.0584, 0.0416, -0.1278},
    {0.0584, 0.1379, 0.0062, -0.1243},
    {0.0416, 0.0062, 0.0936, -0.0393},
    {-0.1278, -0.1243, -0.0393, 0.0},
};
static const double ALPHA0 = 0.95;
static const double AT = -13.152909887;
static const double BT1 = -0.0486839616, BT2 = 3.8, BT3 = 0.62, BT4 = 0.005;
static const double R0_LR = 3.715735, EPS2 = 0.002827918;
static const double LAM1 = 1.338162, LAM2 = 2.260479;
static double EPS1, VSHIFT;
static const double RMR1 = 4.0, RMR2 = 2.9;
static const double AMR0 = -0.2345, AMR1 = -0.67, AMR2 = -4.94;
static int ready = 0;

static double poly5(const double *c, int n, double y) {
    double s = 0.0, p = 1.0;
    for (int i = 0; i < n; i++) {
        s += c[i] * p;
        p *= y;
    }
    return s;
}

static void lut_fill(void);

void lc_init(void) {
    if (ready) return;
    C6 = -(C1 * C1 + 12.0 * C4 * DH * DH) / (30.0 * DH * DH * DH * DH);
    R0H = 1.0 + C1 * DH + 0.5 * (C1 * DH) * (C1 * DH) + C4 * pow(DH, 4) + C6 * pow(DH, 6);
    R1H = C1 + C1 * C1 * DH + 4.0 * C4 * pow(DH, 3) + 6.0 * C6 * pow(DH, 5);
    LH = R0H - 2.0 * C1 * DH;
    KAPPA = (2.0 * C1 - R1H) / LH;
    EPS1 = EPS2 * (LAM2 / LAM1) * (LAM2 / LAM1);
    VSHIFT = EPS1 - EPS2;
    lut_fill();
    ready = 1;
}

static double s_down(double x, double p) {
    if (x <= 0.0) return 1.0;
    if (x >= 1.0) return 0.0;
    double u = 1.0 - x;
    return (1.0 + 2.0 * x + p * x * x) * u * u;
}

static double s_up(double x, double p) { return 1.0 - s_down(x, p); }

static double s_sr(double r) { return s_down((r - 1.7) / 0.5, 3.0); }
static double s_n(double r) { return s_down((r - 1.7) / 0.5, -3.0); }
static double s_lr(double r) { return s_down((r - 5.5) / 0.5, 0.0); }
static double s_mr(double r) { return s_up((r - 1.7) / 0.5, -2.0); }
static double s_up_m(double q) { return s_up((q - 2.0) / 1.0, 0.0); }
static double s_sat(double q) { return s_down((q - 3.0) / 1.0, 0.0); }
static double s_g0(double g) { return s_up((g - 0.34) / 0.59, 0.0); }
static double s_g2(double g) { return s_up((g - 0.30) / 0.63, 0.0); }
static double s_db(double x) { return s_down(x, 0.0); }

/* Tabla radial. 2^20 puntos en 24 MB: la interpolación lineal queda
 * muy por debajo del µeV y el exp() sale del paso de Monte Carlo. */
#define LUT_N (1 << 20)
#define LUT_RMAX LC_CUT
#define LUT_SCALE ((double)(LUT_N - 1) / LUT_RMAX)
static double lut_vr[LUT_N], lut_va[LUT_N], lut_vlr[LUT_N];
static int lut_ready = 0;

static double lut_at(const double *tab, double r) {
    if (r <= 0.0) return tab[0];
    double x = r * LUT_SCALE;
    if (x >= (double)(LUT_N - 1)) return tab[LUT_N - 1];
    int i = (int)x;
    double t = x - (double)i;
    return tab[i] + t * (tab[i + 1] - tab[i]);
}

static double v_r_exact(double r) { return A_SR * exp(-ALPHA * r); }
static double v_a_exact(double r) { return B_SR1 * exp(-BETA1 * r) + B_SR2 * exp(-BETA2 * r); }
static double v_lr_exact(double r);

static double g1(double y) {
    if (y < -0.5) {
        double p = G1C[0] + G1C[1] * y + G1C[2] * y * y;
        double u = y + 1.0;
        return GMIN + u * u * p;
    }
    /* Los g2 de la Tabla I dejan un salto de ~0.05 entre y=-1/2 y y=-1/3.
     * Este cubo hermítico pega G y dG/dy de la rama de la izquierda en
     * y=-1/2 con la de la derecha en y=-1/3. En el grafito sigue valiendo
     * g_gr, y un ángulo que se abre y otro que se cierra suben la energía. */
    if (y < -1.0 / 3.0) {
        double u = y + 0.5;
        return GGR + u * (0.29 + u * (0.60830849577786328 + u * 2.5857763522959947));
    }
    double u = y - 1.0;
    return GMAX + u * u * poly5(G3C, 5, y);
}

static double g1_prim(double y) {
    if (y < -0.5) {
        double p = G1C[0] + G1C[1] * y + G1C[2] * y * y;
        double dp = G1C[1] + 2.0 * G1C[2] * y;
        double u = y + 1.0;
        return 2.0 * u * p + u * u * dp;
    }
    if (y < -1.0 / 3.0) {
        double u = y + 0.5;
        return 0.29 + u * (2.0 * 0.60830849577786328 + 3.0 * u * 2.5857763522959947);
    }
    double r = poly5(G3C, 5, y);
    double dr = G3C[1] + 2.0 * G3C[2] * y + 3.0 * G3C[3] * y * y + 4.0 * G3C[4] * y * y * y;
    double u = y - 1.0;
    return 2.0 * u * r + u * u * dr;
}

static double y0_of(double z) { return AY0 + BY0 * z * (1.0 + z); }

static double g_ang(double y, double z) {
    double yb = y0_of(z);
    if (y <= yb) return g1(y);
    double z4 = z * z * z * z;
    double gzmax = GMAX - (AG + BG * z + CG * z4) * pow(1.0 - yb, 3.0);
    double gz2 = (DG * z4) / (1.0 + EG * z4);
    double dlt = yb - 1.0;
    double d2 = dlt * dlt;
    double d3 = d2 * dlt;
    double gz1 = g1_prim(yb) / d2 - 2.0 * (g1(yb) - gzmax) / d3 - 2.0 * gz2 * yb;
    double gz0 = (g1(yb) - gzmax) / d2 - gz1 * yb - gz2 * yb * yb;
    double uy = 1.0 - y;
    return gzmax + uy * uy * (gz0 + gz1 * y + gz2 * y * y);
}

static double h_of(double x) {
    if (x < -DH) {
        double u = KAPPA * (x + DH);
        return LH * (1.0 + u / pow(1.0 + u * u * u * u, 0.25));
    }
    if (x <= DH) {
        double x2 = x * x;
        return 1.0 + C1 * x + 0.5 * (C1 * x) * (C1 * x) + C4 * x2 * x2 + C6 * x2 * x2 * x2;
    }
    return R0H + R1H * (x - DH);
}

static double v_lr_exact(double r) {
    double s = s_lr(r);
    if (s == 0.0) return 0.0;
    double x = r - R0_LR;
    double m;
    if (r < R0_LR)
        m = EPS1 * (exp(-2.0 * LAM1 * x) - 2.0 * exp(-LAM1 * x)) + VSHIFT;
    else
        m = EPS2 * (exp(-2.0 * LAM2 * x) - 2.0 * exp(-LAM2 * x));
    return s * m;
}

static double v_r(double r) { return lut_ready ? lut_at(lut_vr, r) : v_r_exact(r); }
static double v_a(double r) { return lut_ready ? lut_at(lut_va, r) : v_a_exact(r); }
static double v_lr(double r) { return lut_ready ? lut_at(lut_vlr, r) : v_lr_exact(r); }

static void lut_fill(void) {
    if (lut_ready) return;
    for (int i = 0; i < LUT_N; i++) {
        double r = (double)i / LUT_SCALE;
        lut_vr[i] = v_r_exact(r);
        lut_va[i] = v_a_exact(r);
        lut_vlr[i] = v_lr_exact(r);
    }
    lut_ready = 1;
}

/* x,y van plegadas a una caja. Esto evita nearbyint en el camino caliente. */
static void min_image(double *dx, double *dy, double lx, double ly) {
    double hx = 0.5 * lx;
    double hy = 0.5 * ly;
    if (*dx > hx) *dx -= lx;
    else if (*dx < -hx) *dx += lx;
    if (*dy > hy) *dy -= ly;
    else if (*dy < -hy) *dy += ly;
}

static int bond_geom(const LcView *g, int i, int j, double *dx, double *dy, double *dz, double *r) {
    *dx = g->x[j] - g->x[i];
    *dy = g->y[j] - g->y[i];
    *dz = g->z[j] - g->z[i];
    min_image(dx, dy, g->lx, g->ly);
    double r2 = *dx * *dx + *dy * *dy + *dz * *dz;
    if (r2 >= LC_CUT2 || r2 < 1e-24) return 0;
    *r = sqrt(r2);
    return 1;
}

static const int *nb_of(const LcView *g, int i) { return g->neigh + (size_t)i * LC_MAX_NN; }

static double coord_of(const LcView *g, int i);

#define SR_CAP 12
typedef struct {
    int k;
    double sn, r, dx, dy, dz;
} SrNb;
typedef struct {
    int n;
    int over;
    double coord;
    SrNb v[SR_CAP];
} SrInfo;

static _Thread_local SrInfo *tl_sr;
static _Thread_local int *tl_stamp;
static _Thread_local int tl_sr_cap;
static _Thread_local int tl_era;
static _Thread_local int tl_on;
static _Thread_local const LcView *tl_view;

static const double LC_SR2 = 2.2 * 2.2;

static int sep_r2(const LcView *g, int i, int j, double *dx, double *dy, double *dz, double *r2);

static void sr_fill(const LcView *g, int i, SrInfo *s) {
    const int *nb = nb_of(g, i);
    int m = 0;
    double c = 0.0;
    s->over = 0;
    for (int a = 0; a < g->nn[i]; a++) {
        double dx, dy, dz, r2;
        if (!sep_r2(g, i, nb[a], &dx, &dy, &dz, &r2) || r2 >= LC_SR2) continue;
        double r = sqrt(r2);
        double sn = s_n(r);
        if (sn <= 0.0) continue;
        c += sn;
        if (m < SR_CAP) {
            s->v[m].k = nb[a];
            s->v[m].sn = sn;
            s->v[m].r = r;
            s->v[m].dx = dx;
            s->v[m].dy = dy;
            s->v[m].dz = dz;
            m++;
        } else {
            s->over = 1;
        }
    }
    s->n = m;
    s->coord = c;
}

static SrInfo *sr_get(int i) {
    if (tl_stamp[i] != tl_era) {
        sr_fill(tl_view, i, &tl_sr[i]);
        tl_stamp[i] = tl_era;
    }
    return &tl_sr[i];
}

static void sr_begin(const LcView *g) {
    if (g->n > tl_sr_cap) {
        free(tl_sr);
        free(tl_stamp);
        tl_sr = (SrInfo *)calloc((size_t)g->n, sizeof(SrInfo));
        tl_stamp = (int *)calloc((size_t)g->n, sizeof(int));
        tl_sr_cap = (tl_sr && tl_stamp) ? g->n : 0;
        tl_era = 0;
    }
    if (!tl_sr) {
        tl_on = 0;
        return;
    }
    tl_view = g;
    tl_era++;
    if (tl_era <= 0) {
        memset(tl_stamp, 0, (size_t)tl_sr_cap * sizeof(int));
        tl_era = 1;
    }
    tl_on = 1;
}

static void sr_end(void) { tl_on = 0; }

static double coord_get(const LcView *g, double *coord, int i) {
    if (tl_on) return sr_get(i)->coord;
    if (coord[i] < 0.0) coord[i] = coord_of(g, i);
    return coord[i];
}

static int sep_r2(const LcView *g, int i, int j, double *dx, double *dy, double *dz, double *r2) {
    *dx = g->x[j] - g->x[i];
    *dy = g->y[j] - g->y[i];
    *dz = g->z[j] - g->z[i];
    min_image(dx, dy, g->lx, g->ly);
    *r2 = *dx * *dx + *dy * *dy + *dz * *dz;
    return *r2 < LC_CUT2 && *r2 >= 1e-24;
}

static double coord_of(const LcView *g, int i) {
    const int *nb = nb_of(g, i);
    int ni = g->nn[i];
    double s = 0.0;
    for (int a = 0; a < ni; a++) {
        double dx, dy, dz, r2;
        if (!sep_r2(g, i, nb[a], &dx, &dy, &dz, &r2) || r2 >= LC_SR2) continue;
        s += s_n(sqrt(r2));
    }
    return s;
}

static double b_side(const LcView *g, int i, int j, double ni_coord) {
    if (tl_on) {
        SrInfo *si = sr_get(i);
        if (!si->over) {
            int ij = -1;
            for (int a = 0; a < si->n; a++)
                if (si->v[a].k == j) ij = a;
            if (ij >= 0) {
                double rij = si->v[ij].r;
                double sn_ij = si->v[ij].sn;
                double dx = si->v[ij].dx, dy = si->v[ij].dy, dz = si->v[ij].dz;
                double sum = 0.0;
                for (int a = 0; a < si->n; a++) {
                    if (a == ij || si->v[a].sn == 0.0) continue;
                    double cosv = (dx * si->v[a].dx + dy * si->v[a].dy + dz * si->v[a].dz) / (rij * si->v[a].r);
                    if (cosv > 1.0) cosv = 1.0;
                    if (cosv < -1.0) cosv = -1.0;
                    sum += si->v[a].sn * h_of(rij - si->v[a].r) * g_ang(cosv, si->coord - sn_ij - si->v[a].sn);
                }
                return 1.0 / sqrt(1.0 + sum);
            }
        }
    }
    double dx, dy, dz, rij;
    if (!bond_geom(g, i, j, &dx, &dy, &dz, &rij)) return 1.0;
    double sn_ij = s_n(rij);
    const int *nb = nb_of(g, i);
    int ni = g->nn[i];
    double sum = 0.0;
    for (int a = 0; a < ni; a++) {
        int k = nb[a];
        if (k == j) continue;
        double dxk, dyk, dzk, r2k;
        if (!sep_r2(g, i, k, &dxk, &dyk, &dzk, &r2k) || r2k >= LC_SR2) continue;
        double rik = sqrt(r2k);
        double sn = s_n(rik);
        if (sn == 0.0) continue;
        double cosv = (dx * dxk + dy * dyk + dz * dzk) / (rij * rik);
        if (cosv > 1.0) cosv = 1.0;
        if (cosv < -1.0) cosv = -1.0;
        sum += sn * h_of(rij - rik) * g_ang(cosv, ni_coord - sn_ij - sn);
    }
    return 1.0 / sqrt(1.0 + sum);
}

typedef struct {
    int k;
    double sn, r;
    double dx, dy, dz;
} SideNb;

static int side_list(const LcView *g, int i, int j, SideNb *out) {
    if (tl_on) {
        SrInfo *si = sr_get(i);
        if (!si->over) {
            int m = 0;
            for (int a = 0; a < si->n; a++) {
                if (si->v[a].k == j || si->v[a].sn <= 1e-12) continue;
                if (m >= 8) break;
                out[m].k = si->v[a].k;
                out[m].sn = si->v[a].sn;
                out[m].r = si->v[a].r;
                out[m].dx = si->v[a].dx;
                out[m].dy = si->v[a].dy;
                out[m].dz = si->v[a].dz;
                m++;
            }
            return m;
        }
    }
    const int *nb = nb_of(g, i);
    int m = 0;
    for (int a = 0; a < g->nn[i]; a++) {
        int k = nb[a];
        if (k == j) continue;
        double dx, dy, dz, r2;
        if (!sep_r2(g, i, k, &dx, &dy, &dz, &r2) || r2 >= LC_SR2) continue;
        double r = sqrt(r2);
        double sn = s_n(r);
        if (sn <= 1e-12) continue;
        if (m >= 8) break;
        out[m].k = k;
        out[m].sn = sn;
        out[m].r = r;
        out[m].dx = dx;
        out[m].dy = dy;
        out[m].dz = dz;
        m++;
    }
    return m;
}

static void normalize3(double *x, double *y, double *z) {
    double n = sqrt(*x * *x + *y * *y + *z * *z);
    if (n < 1e-14) return;
    *x /= n;
    *y /= n;
    *z /= n;
}

static int torsion_y(const SideNb *a, int na, unsigned sig, const SideNb *b, int nb, unsigned sigb,
                     double rix, double riy, double riz, double rjx, double rjy, double rjz, double *y) {
    double x1 = 0, y1 = 0, z1 = 0, x2 = 0, y2 = 0, z2 = 0;
    int got = 0;
    for (int t = 0; t < na && got < 2; t++) {
        if (((sig >> t) & 1u) == 0) continue;
        if (got == 0) {
            x1 = a[t].dx;
            y1 = a[t].dy;
            z1 = a[t].dz;
        } else {
            x2 = a[t].dx;
            y2 = a[t].dy;
            z2 = a[t].dz;
        }
        got++;
    }
    if (got < 2) return 0;
    normalize3(&x1, &y1, &z1);
    normalize3(&x2, &y2, &z2);
    double wmx = x1 - x2, wmy = y1 - y2, wmz = z1 - z2;
    double wpx = x1 + x2, wpy = y1 + y2, wpz = z1 + z2;
    normalize3(&wmx, &wmy, &wmz);
    normalize3(&wpx, &wpy, &wpz);
    double tx = riy * wmz - riz * wmy;
    double ty = riz * wmx - rix * wmz;
    double tz = rix * wmy - riy * wmx;
    double dot = rix * wmx + riy * wmy + riz * wmz;
    double cx = riy * wpz - riz * wpy;
    double cy = riz * wpx - rix * wpz;
    double cz = rix * wpy - riy * wpx;
    tx += dot * cx;
    ty += dot * cy;
    tz += dot * cz;
    normalize3(&tx, &ty, &tz);

    got = 0;
    for (int t = 0; t < nb && got < 2; t++) {
        if (((sigb >> t) & 1u) == 0) continue;
        if (got == 0) {
            x1 = b[t].dx;
            y1 = b[t].dy;
            z1 = b[t].dz;
        } else {
            x2 = b[t].dx;
            y2 = b[t].dy;
            z2 = b[t].dz;
        }
        got++;
    }
    if (got < 2) return 0;
    normalize3(&x1, &y1, &z1);
    normalize3(&x2, &y2, &z2);
    wmx = x1 - x2;
    wmy = y1 - y2;
    wmz = z1 - z2;
    wpx = x1 + x2;
    wpy = y1 + y2;
    wpz = z1 + z2;
    normalize3(&wmx, &wmy, &wmz);
    normalize3(&wpx, &wpy, &wpz);
    double ux = rjy * wmz - rjz * wmy;
    double uy = rjz * wmx - rjx * wmz;
    double uz = rjx * wmy - rjy * wmx;
    dot = rjx * wmx + rjy * wmy + rjz * wmz;
    cx = rjy * wpz - rjz * wpy;
    cy = rjz * wpx - rjx * wpz;
    cz = rjx * wpy - rjy * wpx;
    ux += dot * cx;
    uy += dot * cy;
    uz += dot * cz;
    normalize3(&ux, &uy, &uz);
    *y = tx * ux + ty * uy + tz * uz;
    if (*y > 1.0) *y = 1.0;
    if (*y < -1.0) *y = -1.0;
    return 1;
}

static double fat_term(const LcView *g, int i, int j, double *coord) {
    SideNb si[8], sj[8];
    int ni = side_list(g, i, j, si);
    int nj = side_list(g, j, i, sj);
    if (ni > 6 || nj > 6) return 0.0;
    double acc = 0.0;
    int full = 1;
    for (int t = 0; t < ni; t++)
        if (si[t].sn < 1.0 - 1e-9) full = 0;
    for (int t = 0; t < nj; t++)
        if (sj[t].sn < 1.0 - 1e-9) full = 0;
    unsigned nconf = 1u << (ni + nj);
    unsigned conf0 = 0;
    if (full) {
        conf0 = nconf - 1u;
        nconf = conf0 + 1u;
    }
    double dx, dy, dz, rij;
    if (!bond_geom(g, i, j, &dx, &dy, &dz, &rij)) return 0.0;
    double inv = 1.0 / rij;
    double rix = dx * inv, riy = dy * inv, riz = dz * inv;
    for (unsigned conf = conf0; conf < nconf; conf++) {
        unsigned sigi = conf & ((1u << ni) - 1u);
        unsigned sigj = conf >> ni;
        double wi = 1.0;
        int nsig_i = 0;
        double m_i = 0.0;
        for (int t = 0; t < ni; t++) {
            unsigned bit = (sigi >> t) & 1u;
            if (bit) {
                wi *= si[t].sn;
                nsig_i++;
                m_i += s_up_m(coord_get(g, coord, si[t].k) - si[t].sn);
            } else {
                wi *= 1.0 - si[t].sn;
            }
            if (wi == 0.0) break;
        }
        if (wi == 0.0) continue;
        double wj = 1.0;
        int nsig_j = 0;
        double m_j = 0.0;
        for (int t = 0; t < nj; t++) {
            unsigned bit = (sigj >> t) & 1u;
            if (bit) {
                wj *= sj[t].sn;
                nsig_j++;
                m_j += s_up_m(coord_get(g, coord, sj[t].k) - sj[t].sn);
            } else {
                wj *= 1.0 - sj[t].sn;
            }
            if (wj == 0.0) break;
        }
        double w = wi * wj;
        if (w == 0.0) continue;
        if (nsig_i > 3) nsig_i = 3;
        if (nsig_j > 3) nsig_j = 3;
        if (m_i > 3.0) m_i = 3.0;
        if (m_j > 3.0) m_j = 3.0;
        double nel_i = (4.0 - m_i) / (nsig_i + 1.0 - m_i);
        double nel_j = (4.0 - m_j) / (nsig_j + 1.0 - m_j);
        double mn_i = 4.0 / (nsig_i + 1.0);
        double mn_j = 4.0 / (nsig_j + 1.0);
        double mx_i = 4.0 - nsig_i;
        double mx_j = 4.0 - nsig_j;
        double den = mx_i + mx_j - mn_i - mn_j;
        double zc = (fabs(den) < 1e-14) ? 0.0 : (nel_i + nel_j - mn_i - mn_j) / den;
        if (zc < 0.0) zc = 0.0;
        if (zc > 1.0) zc = 1.0;
        acc += w * ((1.0 - zc) * F0[nsig_i][nsig_j] + zc * F1[nsig_i][nsig_j]);
        int anti = (nsig_i == nsig_j && (nsig_i == 1 || nsig_i == 2)) ||
                   (nsig_i == 1 && nsig_j == 2) || (nsig_i == 2 && nsig_j == 1);
        if (anti) {
            double de = nel_i - nel_j;
            acc += w * (ALPHA0 * de * de) / (1.0 + 10.0 * fabs(de));
        }
        if (nsig_i == 2 && nsig_j == 2) {
            double ycos = 0.0;
            if (torsion_y(si, ni, sigi, sj, nj, sigj, rix, riy, riz, -rix, -riy, -riz, &ycos)) {
                double y2 = ycos * ycos;
                double tij;
                if (zc <= 0.125) {
                    double u = zc - 0.125;
                    tij = AT * u * u * (y2 * (1.0 - y2)) * (y2 * (1.0 - y2));
                } else {
                    double de2 = (nel_i - nel_j) * (nel_i - nel_j);
                    double core = (zc - 0.125) * (zc + BT2 * de2 * (de2 - (2.0 / 3.0) * (2.0 / 3.0)));
                    double tau2 = BT1 * core * core * (1.0 - BT3 * zc) / (BT4 + (zc - 0.125) * (zc - 0.125));
                    tij = tau2 * (1.0 - y2) * (2.0 - y2) * (2.0 - y2);
                }
                acc += w * tij;
            }
        }
    }
    return acc;
}

static double gamma_side(const LcView *g, int i, int j, double ni_coord) {
    if (tl_on) {
        SrInfo *si = sr_get(i);
        if (!si->over) {
            int ij = -1;
            for (int a = 0; a < si->n; a++)
                if (si->v[a].k == j) ij = a;
            double rij, dx, dy, dz, sn_ij;
            if (ij >= 0) {
                rij = si->v[ij].r;
                dx = si->v[ij].dx;
                dy = si->v[ij].dy;
                dz = si->v[ij].dz;
                sn_ij = si->v[ij].sn;
            } else if (!bond_geom(g, i, j, &dx, &dy, &dz, &rij)) {
                return 1.0;
            } else {
                sn_ij = s_n(rij);
            }
            double nij = si->coord - sn_ij;
            if (nij < 1e-8) return 1.0;
            double sum = 0.0;
            for (int a = 0; a < si->n; a++) {
                if (a == ij || si->v[a].sn == 0.0) continue;
                double cosv = (dx * si->v[a].dx + dy * si->v[a].dy + dz * si->v[a].dz) / (rij * si->v[a].r);
                if (cosv > 1.0) cosv = 1.0;
                if (cosv < -1.0) cosv = -1.0;
                double u = 1.0 + cosv;
                sum += si->v[a].sn * u * u * u * u;
            }
            return 1.0 / (1.0 + sum / nij);
        }
    }
    double dx, dy, dz, rij;
    if (!bond_geom(g, i, j, &dx, &dy, &dz, &rij)) return 1.0;
    double nij = ni_coord - s_n(rij);
    if (nij < 1e-8) return 1.0;
    const int *nb = nb_of(g, i);
    double sum = 0.0;
    for (int a = 0; a < g->nn[i]; a++) {
        int k = nb[a];
        if (k == j) continue;
        double dxk, dyk, dzk, r2k;
        if (!sep_r2(g, i, k, &dxk, &dyk, &dzk, &r2k) || r2k >= LC_SR2) continue;
        double rik = sqrt(r2k);
        double sn = s_n(rik);
        if (sn == 0.0) continue;
        double cosv = (dx * dxk + dy * dyk + dz * dzk) / (rij * rik);
        if (cosv > 1.0) cosv = 1.0;
        if (cosv < -1.0) cosv = -1.0;
        double u = 1.0 + cosv;
        sum += sn * u * u * u * u;
    }
    /* El orden de enlace del par lejano es << 1 y encendería el alcance medio
     * dentro del grafito, en contra de la Tabla II. Con prefactor 1 el ángulo
     * de 120° deja gamma por debajo de 0.34 y el término se apaga; un enlace
     * que se estira, con ángulos abiertos, lo enciende. */
    return 1.0 / (1.0 + sum / nij);
}

static double nel_of_neighbor(const LcView *g, int k, int i, double sn_ki, double *coord) {
    double nki = coord_get(g, coord, k) - sn_ki;
    const int *nb = nb_of(g, k);
    double m = 0.0;
    for (int a = 0; a < g->nn[k]; a++) {
        int m_id = nb[a];
        if (m_id == i) continue;
        double dx, dy, dz, r2;
        if (!sep_r2(g, k, m_id, &dx, &dy, &dz, &r2) || r2 >= LC_SR2) continue;
        double r = sqrt(r2);
        double sn = s_n(r);
        if (sn == 0.0) continue;
        m += sn * s_up_m(coord_get(g, coord, m_id) - sn);
    }
    double sat = s_sat(nki);
    return (4.0 - sat * m) / (nki + 1.0 - sat * m);
}

static double v_mr_env(const LcView *g, int i, int j, double ni_coord, double *coord) {
    double dx, dy, dz, rij;
    if (!bond_geom(g, i, j, &dx, &dy, &dz, &rij)) return 0.0;
    double sm = s_mr(rij);
    if (sm == 0.0) return 0.0;
    /* Por debajo de 0.30 las dos ventanas del alcance medio valen cero. */
    double gam = gamma_side(g, i, j, ni_coord);
    if (gam <= 0.30) return 0.0;
    const int *nb = nb_of(g, i);
    double ndb = 4.0;
    for (int a = 0; a < g->nn[i]; a++) {
        int k = nb[a];
        if (k == j) continue;
        double dxk, dyk, dzk, r2k;
        if (!sep_r2(g, i, k, &dxk, &dyk, &dzk, &r2k) || r2k >= LC_SR2) continue;
        double rk = sqrt(r2k);
        double sn = s_n(rk);
        if (sn == 0.0) continue;
        ndb -= sn * nel_of_neighbor(g, k, i, sn, coord);
    }
    if (ndb <= 0.0 || ndb > 3.0) return 0.0;
    int base = (int)ndb;
    if (base > 2) base = 2;
    double xdb = ndb - (double)base;
    double vm0 = (rij < RMR1) ? AMR0 * pow(RMR1 - rij, 3.0) : 0.0;
    double vm1 = (rij < RMR1) ? AMR1 * pow(RMR1 - rij, 3.0) : 0.0;
    double vm2 = (rij < RMR2) ? AMR2 * pow(RMR2 - rij, 2.0) : 0.0;
    double vm;
    if (ndb <= 1.0)
        vm = s_db(xdb) * s_g0(gam) * vm0 + (1.0 - s_db(xdb)) * s_g0(gam) * vm1;
    else if (ndb <= 2.0)
        vm = s_db(xdb) * s_g0(gam) * vm1 + (1.0 - s_db(xdb)) * s_g2(gam) * s_g2(gam) * vm2;
    else
        vm = s_g2(gam) * s_g2(gam) * vm2;
    return sm * vm;
}

static double mr_atom(const LcView *g, int i, double *coord) {
    coord_get(g, coord, i);
    const int *nb = nb_of(g, i);
    double sum2 = 0.0;
    for (int a = 0; a < g->nn[i]; a++) {
        double v = v_mr_env(g, i, nb[a], coord[i], coord);
        sum2 += v * v;
    }
    return -sqrt(sum2);
}

static double lc_bond_c(const LcView *g, int i, int j, double *coord);

static _Thread_local double *tl_coord;
static _Thread_local int tl_cap;

static double *coord_scratch(int n) {
    if (n > tl_cap) {
        double *p = (double *)realloc(tl_coord, (size_t)n * sizeof(double));
        if (!p) return NULL;
        tl_coord = p;
        tl_cap = n;
    }
    for (int i = 0; i < n; i++) tl_coord[i] = -1.0;
    return tl_coord;
}

double lc_bond(const LcView *g, int i, int j) {
    if (!ready) lc_init();
    sr_begin(g);
    double *coord = coord_scratch(g->n);
    if (!coord) {
        sr_end();
        return 0.0;
    }
    double e = lc_bond_c(g, i, j, coord);
    sr_end();
    return e;
}

static double lc_bond_c(const LcView *g, int i, int j, double *coord) {
    if (!ready) lc_init();
    double dx, dy, dz, r;
    if (!bond_geom(g, i, j, &dx, &dy, &dz, &r)) return 0.0;
    double e = 0.0;
    double ssr = s_sr(r);
    if (ssr > 0.0) {
        double b = 0.5 * (b_side(g, i, j, coord_get(g, coord, i)) + b_side(g, j, i, coord_get(g, coord, j))) +
                   fat_term(g, i, j, coord);
        e += ssr * (v_r(r) - b * v_a(r));
    }
    if (ssr < 1.0) e += (1.0 - ssr) * v_lr(r);
    return e;
}

void lc_parts(const LcView *g, double out[6]) {
    if (!ready) lc_init();
    for (int k = 0; k < 6; k++) out[k] = 0.0;
    sr_begin(g);
    double *coord = coord_scratch(g->n);
    if (!coord) {
        sr_end();
        return;
    }
    for (int i = 0; i < g->n; i++) {
        const int *nb = nb_of(g, i);
        for (int a = 0; a < g->nn[i]; a++) {
            int j = nb[a];
            if (j < i) continue;
            double dx, dy, dz, r;
            if (!bond_geom(g, i, j, &dx, &dy, &dz, &r)) continue;
            double ssr = s_sr(r);
            if (ssr > 0.0) {
                double bi = b_side(g, i, j, coord_get(g, coord, i));
                double bj = b_side(g, j, i, coord_get(g, coord, j));
                double fat = fat_term(g, i, j, coord);
                double va = v_a(r);
                out[0] += ssr * v_r(r);
                out[1] += ssr * 0.5 * (bi + bj) * va;
                out[2] += ssr * fat * va;
                out[5] += ssr * va;
            }
            if (ssr < 1.0) out[3] += (1.0 - ssr) * v_lr(r);
        }
        out[4] += mr_atom(g, i, coord);
    }
    sr_end();
}

double lc_total(const LcView *g) {
    if (!ready) lc_init();
    sr_begin(g);
    double *coord = coord_scratch(g->n);
    if (!coord) {
        sr_end();
        return 0.0;
    }
    double e = 0.0;
    for (int i = 0; i < g->n; i++) {
        const int *nb = nb_of(g, i);
        for (int a = 0; a < g->nn[i]; a++) {
            int j = nb[a];
            if (j < i) continue;
            e += lc_bond_c(g, i, j, coord);
        }
        e += mr_atom(g, i, coord);
    }
    sr_end();
    return e;
}

static int marked_push(int *ids, int *n, int cap, int v) {
    for (int t = 0; t < *n; t++)
        if (ids[t] == v) return 0;
    if (*n >= cap) return -1;
    ids[(*n)++] = v;
    return 1;
}

static int push_pair(int *pa, int *pb, int *np, int cap, int a, int b) {
    if (a == b) return 0;
    if (a > b) {
        int t = a;
        a = b;
        b = t;
    }
    for (int t = 0; t < *np; t++)
        if (pa[t] == a && pb[t] == b) return 0;
    if (*np >= cap) return -1;
    pa[*np] = a;
    pb[*np] = b;
    (*np)++;
    return 1;
}

/* 2.45 Å = corte 2.20 más 0.25. Vale para un desplazamiento de hasta 0.24 Å,
 * que es más que el paso máximo del Monte Carlo (0.12 Å por eje). */
static int is_sr_link(const LcView *g, int a, int b) {
    double dx, dy, dz, r2;
    if (!sep_r2(g, a, b, &dx, &dy, &dz, &r2)) return 0;
    return r2 < 2.45 * 2.45;
}

int lc_touch_build(const LcView *g, int i, LcTouch *t) {
    if (!ready) lc_init();
    t->np = 0;
    t->na = 0;
    int hop1[32], n1 = 0;
    const int *nbi = nb_of(g, i);
    if (marked_push(t->atoms, &t->na, LC_TOUCH_ATOMS, i) < 0) return -1;
    for (int a = 0; a < g->nn[i]; a++) {
        int j = nbi[a];
        if (push_pair(t->pa, t->pb, &t->np, LC_TOUCH_PAIRS, i, j) < 0) return -1;
        if (!is_sr_link(g, i, j)) continue;
        if (marked_push(hop1, &n1, 32, j) < 0) return -1;
        if (marked_push(t->atoms, &t->na, LC_TOUCH_ATOMS, j) < 0) return -1;
    }
    for (int h = 0; h < n1; h++) {
        int j = hop1[h];
        const int *nbj = nb_of(g, j);
        for (int b = 0; b < g->nn[j]; b++) {
            int k = nbj[b];
            if (!is_sr_link(g, j, k)) continue;
            if (push_pair(t->pa, t->pb, &t->np, LC_TOUCH_PAIRS, j, k) < 0) return -1;
            if (k == i) continue;
            if (marked_push(t->atoms, &t->na, LC_TOUCH_ATOMS, k) < 0) return -1;
            const int *nbk = nb_of(g, k);
            for (int c = 0; c < g->nn[k]; c++) {
                int p = nbk[c];
                if (!is_sr_link(g, k, p)) continue;
                if (push_pair(t->pa, t->pb, &t->np, LC_TOUCH_PAIRS, k, p) < 0) return -1;
            }
        }
    }
    return 0;
}

double lc_touch_eval(const LcView *g, const LcTouch *t) {
    if (!ready) lc_init();
    sr_begin(g);
    double *coord = coord_scratch(g->n);
    if (!coord) {
        sr_end();
        return 0.0;
    }
    double e = 0.0;
    for (int k = 0; k < t->np; k++) e += lc_bond_c(g, t->pa[k], t->pb[k], coord);
    for (int k = 0; k < t->na; k++) e += mr_atom(g, t->atoms[k], coord);
    sr_end();
    return e;
}

double lc_touching(const LcView *g, int i) {
    LcTouch t;
    if (lc_touch_build(g, i, &t) != 0) return 0.0;
    return lc_touch_eval(g, &t);
}

static void sheet(double a, int p, int q, double *x, double *y, double *z, double *lx, double *ly) {
    double a1 = a * sqrt(3.0);
    *lx = p * a1;
    *ly = q * 3.0 * a;
    double bx[4] = {0.0, 0.0, 0.5 * a1, 0.5 * a1};
    double by[4] = {0.0, a, 1.5 * a, 2.5 * a};
    int idx = 0;
    for (int iy = 0; iy < q; iy++) {
        for (int ix = 0; ix < p; ix++) {
            for (int b = 0; b < 4; b++) {
                x[idx] = bx[b] + ix * a1;
                y[idx] = by[b] + iy * 3.0 * a;
                z[idx] = 0.0;
                idx++;
            }
        }
    }
}

static int bind_neighbors(LcView *g, int *nn, int *neigh) {
    int n = g->n;
    memset(nn, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = g->x[j] - g->x[i];
            double dy = g->y[j] - g->y[i];
            double dz = g->z[j] - g->z[i];
            min_image(&dx, &dy, g->lx, g->ly);
            if (dx * dx + dy * dy + dz * dz >= LC_CUT2) continue;
            if (nn[i] >= LC_MAX_NN || nn[j] >= LC_MAX_NN) return -1;
            neigh[(size_t)i * LC_MAX_NN + nn[i]++] = j;
            neigh[(size_t)j * LC_MAX_NN + nn[j]++] = i;
        }
    }
    g->nn = nn;
    g->neigh = neigh;
    return 0;
}

int lc_self_test(void) {
    lc_init();
    const int p = 6, q = 4, n = 4 * p * q;
    double *x = (double *)calloc((size_t)n, sizeof(double));
    double *y = (double *)calloc((size_t)n, sizeof(double));
    double *z = (double *)calloc((size_t)n, sizeof(double));
    int *nn = (int *)calloc((size_t)n, sizeof(int));
    int *neigh = (int *)calloc((size_t)n * LC_MAX_NN, sizeof(int));
    if (!x || !y || !z || !nn || !neigh) return 1;
    double lx, ly;
    sheet(1.42, p, q, x, y, z, &lx, &ly);
    LcView g = {n, lx, ly, x, y, z, nn, neigh};
    int pass = 1;
    if (bind_neighbors(&g, nn, neigh) != 0) pass = 0;
    double e = lc_total(&g);
    double per = e / n;
    printf("lcbopii  N=%d  E/N=%.6f eV  (monocapa de la Tabla II: -7.349)\n", n, per);
    if (fabs(per + 7.349) > 0.005) pass = 0;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        int nsr = 0;
        const int *nb = neigh + (size_t)i * LC_MAX_NN;
        for (int a = 0; a < nn[i]; a++) {
            double dx, dy, dz, r;
            if (!bond_geom(&g, i, nb[a], &dx, &dy, &dz, &r)) continue;
            if (r < 1.7) nsr++;
        }
        if (nsr != 3) bad++;
    }
    if (bad) pass = 0;
    double before = lc_total(&g);
    LcTouch small;
    if (lc_touch_build(&g, 7, &small) != 0) pass = 0;
    double loc0 = lc_touch_eval(&g, &small);
    z[7] = 0.15;
    x[7] += 0.02;
    double after = lc_total(&g);
    double loc1 = lc_touch_eval(&g, &small);
    double d_full = after - before;
    double d_loc = loc1 - loc0;
    printf("lcbopii  dE_full=%.8f  dE_local=%.8f  |diff|=%.3e  nn_sr_malos=%d\n", d_full, d_loc,
           fabs(d_full - d_loc), bad);
    if (fabs(d_full - d_loc) > 1e-6) pass = 0;
    x[7] -= 0.02;
    z[7] = 0.0;
    if (bind_neighbors(&g, nn, neigh) != 0) pass = 0;
    before = lc_total(&g);
    LcTouch region;
    if (lc_touch_build(&g, 7, &region) != 0) pass = 0;
    loc0 = lc_touch_eval(&g, &region);
    x[7] += 0.20;
    if (bind_neighbors(&g, nn, neigh) != 0) pass = 0;
    after = lc_total(&g);
    loc1 = lc_touch_eval(&g, &region);
    printf("lcbopii  dx=0.20 dE_full=%.8f dE_local=%.8f |diff|=%.3e\n", after - before, loc1 - loc0,
           fabs((after - before) - (loc1 - loc0)));
    /* 1e-4: un paso grande puede cruzar la cola de 6 Å, donde V_lr ya es ~10 µeV. */
    if (fabs((after - before) - (loc1 - loc0)) > 1e-4) pass = 0;
    x[7] -= 0.20;
    sheet(1.42, p, q, x, y, z, &lx, &ly);
    g.lx = lx;
    g.ly = ly;
    if (bind_neighbors(&g, nn, neigh) != 0) pass = 0;
    double flat_e = lc_total(&g);
    for (int i = 1; i < n; i += 4) y[i] += 0.02;
    if (bind_neighbors(&g, nn, neigh) != 0) pass = 0;
    double shear_e = lc_total(&g);
    if (!(shear_e > flat_e)) pass = 0;
    printf("lcbopii  cizalla dE/N=%.6f eV\n", (shear_e - flat_e) / n);
    if (pass) printf("lcbopii self-test PASS\n");
    else printf("lcbopii self-test FAIL\n");
    free(x);
    free(y);
    free(z);
    free(nn);
    free(neigh);
    return pass ? 0 : 1;
}
