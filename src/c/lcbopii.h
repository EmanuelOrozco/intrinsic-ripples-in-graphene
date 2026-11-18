/* LCBOPII (Los, Ghiringhelli, Meijer y Fasolino, PRB 72, 214102).
 * La energía de un par no ordenado ya incluye el 1/2 de la doble suma.
 * El Monte Carlo de Tersoff no pasa por aquí. */
#ifndef LCBOPII_H
#define LCBOPII_H

#define LC_MAX_NN 64
#define LC_CUT 6.05
#define LC_CUT2 (LC_CUT * LC_CUT)
#define LC_SR 2.25

typedef struct {
    int n;
    double lx, ly;
    const double *x, *y, *z;
    const int *nn;
    const int *neigh; /* n * LC_MAX_NN, vecinos con r < LC_CUT */
} LcView;

void lc_init(void);

/* Energía del par {i,j} contada una vez. Cero si r >= LC_CUT. */
double lc_bond(const LcView *g, int i, int j);

double lc_total(const LcView *g);

/* rep, atr (b*VA), conj (F+A+T)*VA, largo, medio, suma de S_sr*VA.
 * Energía: rep - atr - conj + largo + medio. */
void lc_parts(const LcView *g, double out[6]);

/* Pares cuya energía puede cambiar si el átomo se mueve menos de 0.24 Å.
 * Se construye una vez y se evalúa antes y después del movimiento. */
#define LC_TOUCH_PAIRS 384
#define LC_TOUCH_ATOMS 160
typedef struct {
    int pa[LC_TOUCH_PAIRS];
    int pb[LC_TOUCH_PAIRS];
    int np;
    int atoms[LC_TOUCH_ATOMS];
    int na;
} LcTouch;

int lc_touch_build(const LcView *g, int i, LcTouch *t);
double lc_touch_eval(const LcView *g, const LcTouch *t);

/* Atajo de una sola geometría. Para un ΔE hay que usar el mismo LcTouch. */
double lc_touching(const LcView *g, int i);

/* Hoja plana: E/N = -7.350 eV frente a 7.349 de la monocapa en la Tabla II,
 * y el ΔE local coincide con el ΔE completo. Devuelve 0 si pasa. */
int lc_self_test(void);

#endif
