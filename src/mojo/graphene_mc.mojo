"""Monte Carlo Metropolis de grafeno (Fasolino et al. 2007) en Mojo.

N=240, T<=300 K. Tersoff C Lindsay–Broido. Una réplica por proceso.
Hot path con Pointer y geometría de vecinos cacheada.
"""

from std.math import sqrt, exp, cos, sin, log, floor, ceil, abs, pow
from std.sys.arg import argv
from std.sys.info import num_logical_cores
from std.time import perf_counter_ns

comptime A_CC = 1.42
comptime KB_EV = 8.617333262145e-5
comptime MAX_NN = 20
comptime NMAX_Q = 10
comptime RDF_BINS = 350
comptime RDF_MAX = 3.5
comptime PI = 3.14159265358979323846
comptime TER_A = 1393.6
comptime TER_B = 430.0
comptime TER_L1 = 3.4879
comptime TER_L2 = 2.2119
comptime TER_BETA = 1.5724e-7
comptime TER_N = 0.72751
comptime TER_C = 38049.0
comptime TER_D = 4.3484
comptime TER_H = -0.93
comptime TER_R = 1.80
comptime TER_S = 2.10
comptime TER_S2 = TER_S * TER_S
comptime TER_C2 = TER_C * TER_C
comptime TER_D2 = TER_D * TER_D
comptime TER_BETA_N = TER_BETA**TER_N
comptime INV_2N = -0.5 / TER_N
comptime TRACE_STRIDE = 200
comptime PROGRESS_STRIDE = 20000


@always_inline
def nint(x: Float64) -> Float64:
    if x >= 0.0:
        return floor(x + 0.5)
    return ceil(x - 0.5)


@always_inline
def wrap(v: Float64, L: Float64) -> Float64:
    return v - L * floor(v / L)


@always_inline
def min_img(d: Float64, L: Float64) -> Float64:
    return d - L * nint(d / L)


@always_inline
def cutoff_fc(r: Float64) -> Float64:
    if r < TER_R:
        return 1.0
    if r >= TER_S:
        return 0.0
    var x = PI * (r - TER_R) / (TER_S - TER_R)
    return 0.5 + 0.5 * cos(x)


@always_inline
def g_theta(cost: Float64) -> Float64:
    var dh = TER_H - cost
    return 1.0 + TER_C2 / TER_D2 - TER_C2 / (TER_D2 + dh * dh)


struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed
        if self.s == 0:
            self.s = 0xA5A5A5A5A5A5A5A5

    @always_inline
    def next_u64(mut self) -> UInt64:
        var x = self.s
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.s = x
        return x

    @always_inline
    def uni(mut self) -> Float64:
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    @always_inline
    def sym(mut self) -> Float64:
        return 2.0 * self.uni() - 1.0


struct Sheet:
    var n: Int
    var lx: Float64
    var ly: Float64
    var x: List[Float64]
    var y: List[Float64]
    var z: List[Float64]
    var nn: List[Int]
    var neigh: List[Int]
    var r2: List[Float64]
    var ndx: List[Float64]
    var ndy: List[Float64]
    var ndz: List[Float64]
    var nr: List[Float64]

    def __init__(out self, p: Int, q: Int):
        var a1 = A_CC * sqrt(3.0)
        self.lx = Float64(p) * a1
        self.ly = Float64(q) * 3.0 * A_CC
        self.n = 4 * p * q
        var n = self.n
        self.x = List[Float64](length=n, fill=0.0)
        self.y = List[Float64](length=n, fill=0.0)
        self.z = List[Float64](length=n, fill=0.0)
        self.nn = List[Int](length=n, fill=0)
        self.neigh = List[Int](length=n * MAX_NN, fill=0)
        self.r2 = List[Float64](length=n * n, fill=0.0)
        self.ndx = List[Float64](length=n * MAX_NN, fill=0.0)
        self.ndy = List[Float64](length=n * MAX_NN, fill=0.0)
        self.ndz = List[Float64](length=n * MAX_NN, fill=0.0)
        self.nr = List[Float64](length=n * MAX_NN, fill=0.0)
        var bx = List[Float64](length=4, fill=0.0)
        var by = List[Float64](length=4, fill=0.0)
        bx[0] = 0.0
        bx[1] = 0.0
        bx[2] = 0.5 * a1
        bx[3] = 0.5 * a1
        by[0] = 0.0
        by[1] = A_CC
        by[2] = 1.5 * A_CC
        by[3] = 2.5 * A_CC
        var idx = 0
        for iy in range(q):
            for ix in range(p):
                for b in range(4):
                    self.x[idx] = bx[b] + Float64(ix) * a1
                    self.y[idx] = by[b] + Float64(iy) * 3.0 * A_CC
                    self.z[idx] = 0.0
                    idx += 1
        self.rebuild_all()

    def rebuild_row(mut self, i: Int):
        var n = self.n
        var px = self.x.unsafe_ptr()
        var py = self.y.unsafe_ptr()
        var pz = self.z.unsafe_ptr()
        var pr2 = self.r2.unsafe_ptr()
        var pnn = self.nn.unsafe_ptr()
        var pnb = self.neigh.unsafe_ptr()
        var pndx = self.ndx.unsafe_ptr()
        var pndy = self.ndy.unsafe_ptr()
        var pndz = self.ndz.unsafe_ptr()
        var pnr = self.nr.unsafe_ptr()
        var lx = self.lx
        var ly = self.ly
        var xi = px[unsafe_offset=i]
        var yi = py[unsafe_offset=i]
        var zi = pz[unsafe_offset=i]
        var ni = 0
        var row = i * n
        for j in range(n):
            if j == i:
                pr2[unsafe_offset=row + i] = 0.0
                continue
            var dx = min_img(px[unsafe_offset=j] - xi, lx)
            var dy = min_img(py[unsafe_offset=j] - yi, ly)
            var dz = pz[unsafe_offset=j] - zi
            var r2v = dx * dx + dy * dy + dz * dz
            pr2[unsafe_offset=row + j] = r2v
            pr2[unsafe_offset=j * n + i] = r2v
            if r2v < TER_S2 and ni < MAX_NN:
                var off = i * MAX_NN + ni
                pnb[unsafe_offset=off] = j
                pndx[unsafe_offset=off] = dx
                pndy[unsafe_offset=off] = dy
                pndz[unsafe_offset=off] = dz
                pnr[unsafe_offset=off] = sqrt(r2v)
                ni += 1
        pnn[unsafe_offset=i] = ni
        for j in range(n):
            if j == i:
                continue
            var nj = pnn[unsafe_offset=j]
            var basej = j * MAX_NN
            var has = False
            var pos = -1
            for a in range(nj):
                if pnb[unsafe_offset=basej + a] == i:
                    has = True
                    pos = a
                    break
            var should = pr2[unsafe_offset=row + j] < TER_S2
            if should:
                var dx = min_img(xi - px[unsafe_offset=j], lx)
                var dy = min_img(yi - py[unsafe_offset=j], ly)
                var dz = zi - pz[unsafe_offset=j]
                var r = sqrt(dx * dx + dy * dy + dz * dz)
                if not has and nj < MAX_NN:
                    var off = basej + nj
                    pnb[unsafe_offset=off] = i
                    pndx[unsafe_offset=off] = dx
                    pndy[unsafe_offset=off] = dy
                    pndz[unsafe_offset=off] = dz
                    pnr[unsafe_offset=off] = r
                    pnn[unsafe_offset=j] = nj + 1
                if has:
                    var off = basej + pos
                    pndx[unsafe_offset=off] = dx
                    pndy[unsafe_offset=off] = dy
                    pndz[unsafe_offset=off] = dz
                    pnr[unsafe_offset=off] = r
            if (not should) and has:
                var last = nj - 1
                var dst = basej + pos
                var src = basej + last
                pnb[unsafe_offset=dst] = pnb[unsafe_offset=src]
                pndx[unsafe_offset=dst] = pndx[unsafe_offset=src]
                pndy[unsafe_offset=dst] = pndy[unsafe_offset=src]
                pndz[unsafe_offset=dst] = pndz[unsafe_offset=src]
                pnr[unsafe_offset=dst] = pnr[unsafe_offset=src]
                pnn[unsafe_offset=j] = last

    def fill_row(mut self, i: Int):
        var n = self.n
        var px = self.x.unsafe_ptr()
        var py = self.y.unsafe_ptr()
        var pz = self.z.unsafe_ptr()
        var pr2 = self.r2.unsafe_ptr()
        var pnn = self.nn.unsafe_ptr()
        var pnb = self.neigh.unsafe_ptr()
        var pndx = self.ndx.unsafe_ptr()
        var pndy = self.ndy.unsafe_ptr()
        var pndz = self.ndz.unsafe_ptr()
        var pnr = self.nr.unsafe_ptr()
        var lx = self.lx
        var ly = self.ly
        var xi = px[unsafe_offset=i]
        var yi = py[unsafe_offset=i]
        var zi = pz[unsafe_offset=i]
        var ni = 0
        var row = i * n
        for j in range(n):
            if j == i:
                pr2[unsafe_offset=row + i] = 0.0
                continue
            var dx = min_img(px[unsafe_offset=j] - xi, lx)
            var dy = min_img(py[unsafe_offset=j] - yi, ly)
            var dz = pz[unsafe_offset=j] - zi
            var r2v = dx * dx + dy * dy + dz * dz
            pr2[unsafe_offset=row + j] = r2v
            if r2v < TER_S2 and ni < MAX_NN:
                var off = i * MAX_NN + ni
                pnb[unsafe_offset=off] = j
                pndx[unsafe_offset=off] = dx
                pndy[unsafe_offset=off] = dy
                pndz[unsafe_offset=off] = dz
                pnr[unsafe_offset=off] = sqrt(r2v)
                ni += 1
        pnn[unsafe_offset=i] = ni

    def rebuild_all(mut self):
        var pnn = self.nn.unsafe_ptr()
        for i in range(self.n):
            pnn[unsafe_offset=i] = 0
        for i in range(self.n):
            self.fill_row(i)

    @always_inline
    def pair_slot(self, i: Int, slot: Int) -> Float64:
        var pndx = self.ndx.unsafe_ptr()
        var pndy = self.ndy.unsafe_ptr()
        var pndz = self.ndz.unsafe_ptr()
        var pnr = self.nr.unsafe_ptr()
        var pnn = self.nn.unsafe_ptr()
        var base = i * MAX_NN
        var dx = pndx[unsafe_offset=base + slot]
        var dy = pndy[unsafe_offset=base + slot]
        var dz = pndz[unsafe_offset=base + slot]
        var rij = pnr[unsafe_offset=base + slot]
        var fc = cutoff_fc(rij)
        if fc == 0.0 or rij < 1e-14:
            return 0.0
        var nn = pnn[unsafe_offset=i]
        var zeta: Float64 = 0.0
        for a in range(nn):
            if a == slot:
                continue
            var dxk = pndx[unsafe_offset=base + a]
            var dyk = pndy[unsafe_offset=base + a]
            var dzk = pndz[unsafe_offset=base + a]
            var rik = pnr[unsafe_offset=base + a]
            var fck = cutoff_fc(rik)
            if fck == 0.0 or rik < 1e-14:
                continue
            var cost = (dx * dxk + dy * dyk + dz * dzk) / (rij * rik)
            if cost > 1.0:
                cost = 1.0
            if cost < -1.0:
                cost = -1.0
            zeta += fck * g_theta(cost)
        var zn: Float64 = 0.0
        if zeta > 0.0:
            zn = pow(zeta, TER_N)
        var b = pow(1.0 + TER_BETA_N * zn, INV_2N)
        var fR = TER_A * exp(-TER_L1 * rij)
        var fA = -TER_B * exp(-TER_L2 * rij)
        return fc * (fR + b * fA)

    def local_energy(self, i: Int) -> Float64:
        var pnb = self.neigh.unsafe_ptr()
        var pnn = self.nn.unsafe_ptr()
        var e: Float64 = 0.0
        var ni = pnn[unsafe_offset=i]
        var basei = i * MAX_NN
        for a in range(ni):
            var j = pnb[unsafe_offset=basei + a]
            e += 0.5 * self.pair_slot(i, a)
            var nj = pnn[unsafe_offset=j]
            var basej = j * MAX_NN
            var slot_i = 0
            for b in range(nj):
                if pnb[unsafe_offset=basej + b] == i:
                    slot_i = b
                    break
            e += 0.5 * self.pair_slot(j, slot_i)
            for b in range(nj):
                if pnb[unsafe_offset=basej + b] == i:
                    continue
                e += 0.5 * self.pair_slot(j, b)
        return e

    def full_energy(self) -> Float64:
        var pnn = self.nn.unsafe_ptr()
        var e: Float64 = 0.0
        for i in range(self.n):
            var ni = pnn[unsafe_offset=i]
            for a in range(ni):
                e += 0.5 * self.pair_slot(i, a)
        return e

    def h_rms(self) -> Float64:
        var pz = self.z.unsafe_ptr()
        var n = self.n
        var zmean: Float64 = 0.0
        for i in range(n):
            zmean += pz[unsafe_offset=i]
        zmean = zmean / Float64(n)
        var h2: Float64 = 0.0
        for i in range(n):
            var hi = pz[unsafe_offset=i] - zmean
            h2 += hi * hi
        return sqrt(h2 / Float64(n))


def pad2(n: Int) -> String:
    if n < 10:
        return "0" + String(n)
    return String(n)


def accumulate(
    s: Sheet,
    mut gq: List[Float64],
    mut rdf: List[Float64],
    mut h_acc: Float64,
    mut nsamp: Int,
):
    var n = s.n
    var px = s.x.unsafe_ptr()
    var py = s.y.unsafe_ptr()
    var pz = s.z.unsafe_ptr()
    var zmean: Float64 = 0.0
    for i in range(n):
        zmean += pz[unsafe_offset=i]
    zmean = zmean / Float64(n)
    var h = List[Float64](length=n, fill=0.0)
    var ph = h.unsafe_ptr()
    var h2: Float64 = 0.0
    for i in range(n):
        var hi = pz[unsafe_offset=i] - zmean
        ph[unsafe_offset=i] = hi
        h2 += hi * hi
    h_acc += sqrt(h2 / Float64(n))
    nsamp += 1

    var nmax = NMAX_Q
    var idx = 0
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            if nx == 0 and ny == 0:
                idx += 1
                continue
            var qx = 2.0 * PI * Float64(nx) / s.lx
            var qy = 2.0 * PI * Float64(ny) / s.ly
            var q2 = qx * qx + qy * qy
            var hr: Float64 = 0.0
            var hi: Float64 = 0.0
            for j in range(n):
                var phase = qx * px[unsafe_offset=j] + qy * py[unsafe_offset=j]
                hr += ph[unsafe_offset=j] * cos(phase)
                hi -= ph[unsafe_offset=j] * sin(phase)
            gq[idx] += q2 * (hr * hr + hi * hi) / Float64(n)
            idx += 1

    var dr = RDF_MAX / Float64(RDF_BINS)
    for i in range(n):
        for j in range(i + 1, n):
            var dx = min_img(px[unsafe_offset=j] - px[unsafe_offset=i], s.lx)
            var dy = min_img(py[unsafe_offset=j] - py[unsafe_offset=i], s.ly)
            var dz = pz[unsafe_offset=j] - pz[unsafe_offset=i]
            var r = sqrt(dx * dx + dy * dy + dz * dz)
            if r >= RDF_MAX:
                continue
            var b = Int(r / dr)
            if b >= 0 and b < RDF_BINS:
                rdf[b] += 2.0


def write_outputs(
    replica: Int,
    outdir: String,
    s: Sheet,
    energy: Float64,
    hrms: Float64,
    heq: Float64,
    acc_move: Float64,
    acc_area: Float64,
    nsamp: Int,
    step: Float64,
    T: Float64,
    n_eq: Int,
    n_prod: Int,
    gq: List[Float64],
    rdf: List[Float64],
    tr_step: List[Int],
    tr_e: List[Float64],
    tr_h: List[Float64],
    tr_lx: List[Float64],
    tr_ly: List[Float64],
    tr_am: List[Float64],
    tr_aa: List[Float64],
    ntrace: Int,
) raises:
    var tag = pad2(replica)
    var n = s.n
    var nmax = NMAX_Q
    with open(outdir + "/replica_" + tag + "_summary.csv", "w") as f:
        f.write(
            "replica,N,T,n_eq,n_prod,energy,h_rms,h_rms_eq,lx,ly,S0,acc_move,acc_area,nsamples,step_amp\n"
        )
        f.write(
            String(replica)
            + ","
            + String(n)
            + ","
            + String(T)
            + ","
            + String(n_eq)
            + ","
            + String(n_prod)
            + ","
            + String(energy)
            + ","
            + String(hrms)
            + ","
            + String(heq)
            + ","
            + String(s.lx)
            + ","
            + String(s.ly)
            + ","
            + String(s.lx * s.ly / Float64(n))
            + ","
            + String(acc_move)
            + ","
            + String(acc_area)
            + ","
            + String(nsamp)
            + ","
            + String(step)
            + "\n"
        )

    with open(outdir + "/replica_" + tag + "_gq.csv", "w") as f:
        f.write("nx,ny,q,G_over_N\n")
        var idx = 0
        var inv = 1.0
        if nsamp > 0:
            inv = 1.0 / Float64(nsamp)
        for nx in range(-nmax, nmax + 1):
            for ny in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0:
                    idx += 1
                    continue
                var qx = 2.0 * PI * Float64(nx) / s.lx
                var qy = 2.0 * PI * Float64(ny) / s.ly
                var q = sqrt(qx * qx + qy * qy)
                f.write(
                    String(nx)
                    + ","
                    + String(ny)
                    + ","
                    + String(q)
                    + ","
                    + String(gq[idx] * inv)
                    + "\n"
                )
                idx += 1

    with open(outdir + "/replica_" + tag + "_rdf.csv", "w") as f:
        f.write("r,g\n")
        var density = Float64(n) / (s.lx * s.ly)
        var dr = RDF_MAX / Float64(RDF_BINS)
        for b in range(RDF_BINS):
            var r = (Float64(b) + 0.5) * dr
            var shell = 2.0 * PI * r * dr * density
            var g: Float64 = 0.0
            if nsamp > 0 and shell > 0.0:
                g = rdf[b] / (Float64(nsamp) * Float64(n) * shell)
            f.write(String(r) + "," + String(g) + "\n")

    with open(outdir + "/replica_" + tag + "_snapshot.csv", "w") as f:
        f.write("x,y,z,lx,ly\n")
        var px = s.x.unsafe_ptr()
        var py = s.y.unsafe_ptr()
        var pz = s.z.unsafe_ptr()
        for i in range(n):
            f.write(
                String(px[unsafe_offset=i])
                + ","
                + String(py[unsafe_offset=i])
                + ","
                + String(pz[unsafe_offset=i])
                + ","
                + String(s.lx)
                + ","
                + String(s.ly)
                + "\n"
            )

    with open(outdir + "/replica_" + tag + "_trace.csv", "w") as f:
        f.write("step,energy,h_rms,lx,ly,acc_move,acc_area\n")
        for t in range(ntrace):
            f.write(
                String(tr_step[t])
                + ","
                + String(tr_e[t])
                + ","
                + String(tr_h[t])
                + ","
                + String(tr_lx[t])
                + ","
                + String(tr_ly[t])
                + ","
                + String(tr_am[t])
                + ","
                + String(tr_aa[t])
                + "\n"
            )


def run_replica(
    replica: Int,
    p: Int,
    q: Int,
    T: Float64,
    n_eq: Int,
    n_prod: Int,
    sample_every: Int,
    seed: UInt64,
    outdir: String,
) raises:
    var s = Sheet(p, q)
    var n = s.n
    var rng = Rng(seed + UInt64(replica) * 0xD1B54A32D192ED03)
    var beta = 1.0 / (KB_EV * T)
    var step: Float64 = 0.04
    var area_step: Float64 = 0.002
    var energy = s.full_energy()
    var nmodes = (2 * NMAX_Q + 1) * (2 * NMAX_Q + 1)
    var gq = List[Float64](length=nmodes, fill=0.0)
    var rdf = List[Float64](length=RDF_BINS, fill=0.0)
    var h_acc: Float64 = 0.0
    var nsamp = 0
    var acc_move: Int = 0
    var try_move: Int = 0
    var acc_area: Int = 0
    var try_area: Int = 0
    var heq: Float64 = 0.0
    var total = n_eq + n_prod
    var tcap = total / TRACE_STRIDE + 8
    var tr_step = List[Int](length=tcap, fill=0)
    var tr_e = List[Float64](length=tcap, fill=0.0)
    var tr_h = List[Float64](length=tcap, fill=0.0)
    var tr_lx = List[Float64](length=tcap, fill=0.0)
    var tr_ly = List[Float64](length=tcap, fill=0.0)
    var tr_am = List[Float64](length=tcap, fill=0.0)
    var tr_aa = List[Float64](length=tcap, fill=0.0)
    var ntrace = 0
    var xs = List[Float64](length=n, fill=0.0)
    var ys = List[Float64](length=n, fill=0.0)
    var t0 = perf_counter_ns()
    var px = s.x.unsafe_ptr()
    var py = s.y.unsafe_ptr()
    var pz = s.z.unsafe_ptr()
    var pxs = xs.unsafe_ptr()
    var pys = ys.unsafe_ptr()

    var istep = 1
    while istep <= total:
        var att = 0
        while att < n:
            var i = Int(rng.uni() * Float64(n))
            if i >= n:
                i = n - 1
            var e_old = s.local_energy(i)
            var ox = px[unsafe_offset=i]
            var oy = py[unsafe_offset=i]
            var oz = pz[unsafe_offset=i]
            px[unsafe_offset=i] = wrap(ox + step * rng.sym(), s.lx)
            py[unsafe_offset=i] = wrap(oy + step * rng.sym(), s.ly)
            pz[unsafe_offset=i] = oz + step * rng.sym()
            s.rebuild_row(i)
            var e_new = s.local_energy(i)
            var dE = e_new - e_old
            try_move += 1
            var ok = dE <= 0.0
            if not ok:
                ok = rng.uni() < exp(-beta * dE)
            if ok:
                energy += dE
                acc_move += 1
            else:
                px[unsafe_offset=i] = ox
                py[unsafe_offset=i] = oy
                pz[unsafe_offset=i] = oz
                s.rebuild_row(i)
            att += 1

        if istep % 5 == 0:
            var factor = exp(area_step * rng.sym())
            var lx_n = s.lx * factor
            var ly_n = s.ly * factor
            for i in range(n):
                pxs[unsafe_offset=i] = px[unsafe_offset=i]
                pys[unsafe_offset=i] = py[unsafe_offset=i]
                px[unsafe_offset=i] = px[unsafe_offset=i] * factor
                py[unsafe_offset=i] = py[unsafe_offset=i] * factor
            var old_lx = s.lx
            var old_ly = s.ly
            s.lx = lx_n
            s.ly = ly_n
            s.rebuild_all()
            var e_new = s.full_energy()
            var dE = e_new - energy
            var dlogA = 2.0 * log(factor)
            var acc_p = exp(-beta * dE + Float64(n) * dlogA)
            try_area += 1
            if rng.uni() < acc_p:
                energy = e_new
                acc_area += 1
            else:
                for i in range(n):
                    px[unsafe_offset=i] = pxs[unsafe_offset=i]
                    py[unsafe_offset=i] = pys[unsafe_offset=i]
                s.lx = old_lx
                s.ly = old_ly
                s.rebuild_all()

        if istep == n_eq:
            heq = s.h_rms()

        if istep > n_eq and sample_every > 0 and (istep - n_eq) % sample_every == 0:
            accumulate(s, gq, rdf, h_acc, nsamp)

        if istep % TRACE_STRIDE == 0 and ntrace < tcap:
            var am: Float64 = 0.0
            var aa: Float64 = 0.0
            if try_move > 0:
                am = Float64(acc_move) / Float64(try_move)
            if try_area > 0:
                aa = Float64(acc_area) / Float64(try_area)
            tr_step[ntrace] = istep
            tr_e[ntrace] = energy
            tr_h[ntrace] = s.h_rms()
            tr_lx[ntrace] = s.lx
            tr_ly[ntrace] = s.ly
            tr_am[ntrace] = am
            tr_aa[ntrace] = aa
            ntrace += 1

        if istep % 400 == 0 and try_move > 0:
            var ar = Float64(acc_move) / Float64(try_move)
            if ar > 0.55:
                step = step * 1.08
            if ar < 0.35:
                step = step * 0.92
            if step < 0.005:
                step = 0.005
            if step > 0.12:
                step = 0.12

        if istep % PROGRESS_STRIDE == 0:
            var dt = Float64(perf_counter_ns() - t0) * 1e-9
            print(
                "replica",
                replica,
                "step",
                istep,
                "/",
                total,
                "h_rms",
                s.h_rms(),
                "t",
                dt,
                "s",
            )

        istep += 1

    var am: Float64 = 0.0
    var aa: Float64 = 0.0
    if try_move > 0:
        am = Float64(acc_move) / Float64(try_move)
    if try_area > 0:
        aa = Float64(acc_area) / Float64(try_area)
    var hprod = s.h_rms()
    if nsamp > 0:
        hprod = h_acc / Float64(nsamp)
    write_outputs(
        replica,
        outdir,
        s,
        energy,
        hprod,
        heq,
        am,
        aa,
        nsamp,
        step,
        T,
        n_eq,
        n_prod,
        gq,
        rdf,
        tr_step,
        tr_e,
        tr_h,
        tr_lx,
        tr_ly,
        tr_am,
        tr_aa,
        ntrace,
    )
    var dt = Float64(perf_counter_ns() - t0) * 1e-9
    print(
        "replica",
        pad2(replica),
        "E=",
        energy,
        "h_rms=",
        hprod,
        "acc=",
        am,
        "samples=",
        nsamp,
        "t=",
        dt,
        "s",
    )


def self_test() raises -> Int:
    var s = Sheet(10, 6)
    var e0 = s.full_energy()
    var i = 7
    var loc0 = s.local_energy(i)
    s.x[i] += 0.03
    s.y[i] -= 0.02
    s.z[i] += 0.04
    s.rebuild_row(i)
    var e1 = s.full_energy()
    var loc1 = s.local_energy(i)
    var err = abs((e1 - e0) - (loc1 - loc0))
    print("self-test  N=", s.n, " Lx=", s.lx, " Ly=", s.ly, " E/N=", e0 / Float64(s.n))
    print("self-test  dE_full=", e1 - e0, " dE_local=", loc1 - loc0, " |diff|=", err)
    if s.n != 240:
        return 1
    if abs(s.lx - 24.59) > 0.05:
        return 1
    if abs(s.ly - 25.56) > 0.05:
        return 1
    if e0 / Float64(s.n) > -4.0:
        return 1
    if err > 1e-6:
        return 1
    print("self-test PASS")
    return 0


def usage():
    print("graphene_mc [--T 300] [--n-eq 400000] [--n-prod 500000] [--replica 0]")
    print("            [--sample-every 10] [--seed 42] [--out DIR] [--self-test]")


def main() raises:
    var T: Float64 = 300.0
    var n_eq = 400000
    var n_prod = 500000
    var sample_every = 10
    var replica = 0
    var seed: UInt64 = 42
    var outdir = String("../../results/datos")
    var p = 10
    var q = 6
    var do_test = False
    var args = argv()
    var i = 1
    while i < len(args):
        var a = args[i]
        if a == "--T" and i + 1 < len(args):
            i += 1
            T = atof(args[i])
        elif a == "--n-eq" and i + 1 < len(args):
            i += 1
            n_eq = atol(args[i])
        elif a == "--n-prod" and i + 1 < len(args):
            i += 1
            n_prod = atol(args[i])
        elif a == "--sample-every" and i + 1 < len(args):
            i += 1
            sample_every = atol(args[i])
        elif a == "--replica" and i + 1 < len(args):
            i += 1
            replica = atol(args[i])
        elif a == "--seed" and i + 1 < len(args):
            i += 1
            seed = UInt64(atol(args[i]))
        elif a == "--out" and i + 1 < len(args):
            i += 1
            outdir = String(args[i])
        elif a == "--p" and i + 1 < len(args):
            i += 1
            p = atol(args[i])
        elif a == "--q" and i + 1 < len(args):
            i += 1
            q = atol(args[i])
        elif a == "--self-test":
            do_test = True
        elif a == "--help":
            usage()
            return
        i += 1

    if do_test:
        var rc = self_test()
        if rc != 0:
            print("self-test FAIL")
        return

    if T > 300.0 + 1e-9:
        print("T máxima permitida: 300 K")
        return

    print(
        "Grafeno MC Mojo  N=",
        4 * p * q,
        " T=",
        T,
        " replica=",
        replica,
        " cores_logicos=",
        num_logical_cores(),
    )
    run_replica(replica, p, q, T, n_eq, n_prod, sample_every, seed, outdir)
