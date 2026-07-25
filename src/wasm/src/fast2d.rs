//! Spotifytrack-specialized 2D derivative computer.
//!
//! Key differences from the original port in lib.rs:
//!  - The Hessian is never materialized.  `compute_step_size` only ever needs the scalar
//!    g'Hg, and H has Laplacian structure (H[u][v]=idk, H[u][u]=-sum idk), so
//!    g'Hg = -sum_{u<v} idk_uv * (g[u]-g[v])^2 per dimension.
//!  - The P-stress skip condition ((dist > ideal) && (weight > 1)) is prebaked: non-adjacent
//!    connected pairs store their ideal distance in a packed upper-triangle array (`tri`);
//!    adjacent pairs live in a static edge list; disconnected pairs store 0 (never active).
//!    The dense sweep is just dist^2 < ideal^2 per pair; sqrt/div only run for active pairs.
//!  - Symmetry: pairs are visited once (u<v) with +/- gradient updates.
//!  - The whole rungeKutta integrator runs in wasm; positions live in wasm memory.

use rand::prelude::*;
use rand_pcg::Pcg32;
use std::arch::wasm32::*;
use wasm_bindgen::prelude::*;

const DISP_EPS: f32 = 0.000000001;
const DISP_EPS_SQ: f32 = DISP_EPS * DISP_EPS;

#[derive(Clone, Copy)]
struct Edge {
    u: u32,
    v: u32,
    ideal: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ActivePair {
    u: u16,
    v: u16,
    idk_x: f32,
    idk_y: f32,
}

pub struct FastCtx {
    n: usize,
    /// Packed upper triangle (u<v): ideal distance for swept (non-adjacent, connected) pairs,
    /// 0 for pairs that are never swept (edges, disconnected)
    tri: Vec<f32>,
    row_off: Vec<usize>,
    edges: Vec<Edge>,
    /// false until a non-empty G is set; before that all connected pairs are active (weight=1 mode)
    p_stress: bool,
    x: Vec<f32>,
    g: Vec<f32>,
    active: Vec<ActivePair>,
    huu: Vec<f32>,
    track_huu: bool,
    max_h: f32,
    locks: Vec<[f32; 3]>,
    lock_us: Vec<u32>,
    rng: Pcg32,
    min_d: f32,
    snapshot: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    ia: Vec<f32>,
    ib: Vec<f32>,
}

impl FastCtx {
    pub fn new(d_mat: Vec<f32>, n: usize) -> Self {
        assert!(n < u16::MAX as usize, "fast2d supports at most 65534 nodes");
        let tri_len = n * (n - 1) / 2;
        let mut tri = Vec::with_capacity(tri_len);
        let mut row_off = Vec::with_capacity(n);
        let mut off = 0usize;
        for u in 0..n {
            row_off.push(off.wrapping_sub(u + 1));
            for v in (u + 1)..n {
                let d = d_mat[u * n + v];
                tri.push(if d.is_finite() && d > 0. { d } else { 0. });
                off += 1;
            }
        }

        let mut min_d = d_mat
            .iter()
            .copied()
            .filter(|&x| !x.is_nan() && !x.is_infinite() && x > 0.)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f32::MAX);
        if min_d == f32::MAX {
            min_d = 1.;
        }

        FastCtx {
            n,
            tri,
            row_off,
            edges: Vec::new(),
            p_stress: false,
            x: vec![0.; 2 * n],
            g: vec![0.; 2 * n],
            active: Vec::with_capacity(tri_len),
            huu: vec![0.; 2 * n],
            track_huu: false,
            max_h: 0.,
            locks: Vec::new(),
            lock_us: Vec::new(),
            rng: Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7),
            min_d,
            snapshot: vec![0.; 2 * n],
            a: vec![0.; 2 * n],
            b: vec![0.; 2 * n],
            c: vec![0.; 2 * n],
            d: vec![0.; 2 * n],
            ia: vec![0.; 2 * n],
            ib: vec![0.; 2 * n],
        }
    }

    /// A non-empty G flips into p-stress mode: pairs with weight <= 1 become the static edge
    /// list, everything else is swept with the dist < ideal condition.  An empty G restores
    /// weight=1 mode (everything connected is active).
    pub fn set_g(&mut self, g_mat: Vec<f32>) {
        for e in self.edges.drain(..) {
            let (u, v) = (e.u as usize, e.v as usize);
            self.tri[self.row_off[u].wrapping_add(v)] = e.ideal;
        }
        if g_mat.is_empty() {
            self.p_stress = false;
            return;
        }

        let n = self.n;
        for u in 0..n {
            let base = self.row_off[u];
            for v in (u + 1)..n {
                let ideal = self.tri[base.wrapping_add(v)];
                if ideal > 0. && g_mat[u * n + v] <= 1. {
                    self.edges.push(Edge {
                        u: u as u32,
                        v: v as u32,
                        ideal,
                    });
                    self.tri[base.wrapping_add(v)] = 0.;
                }
            }
        }
        self.p_stress = true;
    }

    fn offset_dir(&mut self) -> [f32; 2] {
        let mut u = [0.; 2];
        let mut l = 0.;
        for i in 0..2 {
            let x: f32 = self.rng.gen_range(0.01, 1.) - 0.5;
            u[i] = x;
            l += x * x;
        }
        l = l.sqrt();
        for x in &mut u {
            *x *= self.min_d / l;
        }
        u
    }

    /// Sweeps all pairs: accumulates the gradient and the active pair list, returns the min
    /// squared distance seen across all u<v pairs (for coincidence detection).
    fn sweep(&mut self, x: *const f32) -> f32 {
        let n = self.n;
        self.g.fill(0.);
        self.lock_us.clear();
        if self.track_huu {
            self.huu.fill(0.);
        }

        let gx = self.g.as_mut_ptr();
        let gy = unsafe { gx.add(n) };
        let hx = self.huu.as_mut_ptr();
        let hy = unsafe { hx.add(n) };
        let xs = x;
        let ys = unsafe { xs.add(n) };
        let ap = self.active.as_mut_ptr();
        let mut alen = 0usize;
        let track_huu = self.track_huu;
        let mut min_d2 = f32::INFINITY;

        macro_rules! process_active {
            ($u:expr, $v:expr, $dx:expr, $dy:expr, $d2:expr, $s:expr, $gxu:expr, $gyu:expr) => {{
                let dxl = $dx;
                let dyl = $dy;
                let sl = $s;
                let vv = $v;
                let d = $d2.sqrt();
                let s2 = sl * sl;
                let rid2 = 1. / s2;
                let r = 1. / d;
                let gs = 2. * (d - sl) * rid2 * r;
                let gxc = dxl * gs;
                let gyc = dyl * gs;
                *$gxu += gxc;
                *$gyu += gyc;
                unsafe {
                    *gx.add(vv) -= gxc;
                    *gy.add(vv) -= gyc;
                }
                let r3 = r * r * r;
                let cc = 2. * rid2 * sl * r3;
                let f4 = 4. * rid2;
                let idk_x = cc * dyl * dyl - f4;
                let idk_y = cc * dxl * dxl - f4;
                unsafe {
                    ap.add(alen).write(ActivePair {
                        u: $u as u16,
                        v: vv as u16,
                        idk_x,
                        idk_y,
                    });
                }
                alen += 1;
                if track_huu {
                    unsafe {
                        *hx.add($u) -= idk_x;
                        *hx.add(vv) -= idk_x;
                        *hy.add($u) -= idk_y;
                        *hy.add(vv) -= idk_y;
                    }
                }
            }};
        }

        for ei in 0..self.edges.len() {
            let e = unsafe { *self.edges.get_unchecked(ei) };
            let (u, v) = (e.u as usize, e.v as usize);
            unsafe {
                let dx = *xs.add(u) - *xs.add(v);
                let dy = *ys.add(u) - *ys.add(v);
                let d2 = dx * dx + dy * dy;
                if d2 < min_d2 {
                    min_d2 = d2;
                }
                if d2 <= DISP_EPS_SQ {
                    continue;
                }
                process_active!(u, v, dx, dy, d2, e.ideal, gx.add(u), gy.add(u));
            }
        }

        let tri = self.tri.as_ptr();
        let p_stress = self.p_stress;
        let mut min_v = f32x4_splat(f32::INFINITY);
        let zero = f32x4_splat(0.);

        for u in 0..n {
            let row = self.row_off[u];
            let vbase = u + 1;
            let len = n - vbase;
            let chunks = len / 4;

            let mut gxu = 0f32;
            let mut gyu = 0f32;

            unsafe {
                let xu = f32x4_splat(*xs.add(u));
                let yu = f32x4_splat(*ys.add(u));

                for ci in 0..chunks {
                    let vi = vbase + ci * 4;
                    let xv = v128_load(xs.add(vi) as *const v128);
                    let yv = v128_load(ys.add(vi) as *const v128);
                    let dx = f32x4_sub(xu, xv);
                    let dy = f32x4_sub(yu, yv);
                    let d2 = f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy));
                    min_v = f32x4_pmin(min_v, d2);
                    let s = v128_load(tri.add(row.wrapping_add(vi)) as *const v128);
                    let mask = if p_stress {
                        f32x4_lt(d2, f32x4_mul(s, s))
                    } else {
                        f32x4_gt(s, zero)
                    };
                    let bits = i32x4_bitmask(mask) as u32;
                    if bits == 0 {
                        continue;
                    }

                    macro_rules! lane {
                        ($L:literal) => {
                            if bits & (1 << $L) != 0 {
                                let d2l = f32x4_extract_lane::<$L>(d2);
                                if d2l > DISP_EPS_SQ {
                                    process_active!(
                                        u,
                                        vi + $L,
                                        f32x4_extract_lane::<$L>(dx),
                                        f32x4_extract_lane::<$L>(dy),
                                        d2l,
                                        f32x4_extract_lane::<$L>(s),
                                        &mut gxu,
                                        &mut gyu
                                    );
                                }
                            }
                        };
                    }
                    lane!(0);
                    lane!(1);
                    lane!(2);
                    lane!(3);
                }

                for v in (vbase + chunks * 4)..n {
                    let dx = *xs.add(u) - *xs.add(v);
                    let dy = *ys.add(u) - *ys.add(v);
                    let d2 = dx * dx + dy * dy;
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                    let sl = *tri.add(row.wrapping_add(v));
                    let is_active = if p_stress { d2 < sl * sl } else { sl > 0. };
                    if is_active && d2 > DISP_EPS_SQ {
                        process_active!(u, v, dx, dy, d2, sl, &mut gxu, &mut gyu);
                    }
                }

                *gx.add(u) += gxu;
                *gy.add(u) += gyu;
            }
        }

        unsafe { self.active.set_len(alen) };

        let mv = f32x4_pmin(
            min_v,
            i32x4_shuffle::<2, 3, 0, 1>(min_v, min_v),
        );
        let mv = f32x4_pmin(mv, i32x4_shuffle::<1, 0, 3, 2>(mv, mv));
        let simd_min = f32x4_extract_lane::<0>(mv);
        if simd_min < min_d2 {
            min_d2 = simd_min;
        }
        min_d2
    }

    /// Same semantics as the original: displacement decisions are made against positions as
    /// they were when distances were computed, while displacements mutate live positions.
    fn apply_displacements(&mut self, x: *mut f32) -> bool {
        let n = self.n;
        unsafe {
            std::ptr::copy_nonoverlapping(x, self.snapshot.as_mut_ptr(), 2 * n);
        }
        let mut did_apply = false;
        for u in 0..n {
            for v in 0..n {
                if u == v {
                    continue;
                }
                let dx = self.snapshot[u] - self.snapshot[v];
                let dy = self.snapshot[n + u] - self.snapshot[n + v];
                if (dx * dx + dy * dy).sqrt() > DISP_EPS {
                    continue;
                }
                did_apply = true;
                let rd = self.offset_dir();
                unsafe {
                    *x.add(v) += rd[0];
                    *x.add(n + v) += rd[1];
                }
            }
        }
        did_apply
    }

    pub fn compute(&mut self, x: *mut f32) {
        loop {
            let min_d2 = self.sweep(x);
            if min_d2 > DISP_EPS_SQ {
                break;
            }
            if !self.apply_displacements(x) {
                break;
            }
        }

        if self.track_huu {
            let mut max_h = 0f32;
            for &h in &self.huu {
                if h > max_h {
                    max_h = h;
                }
            }
            self.max_h = max_h;
        }
    }

    pub fn apply_lock(&mut self, u: usize, p: [f32; 2], x_u: [f32; 2]) {
        let n = self.n;
        self.g[u] -= self.max_h * (p[0] - x_u[0]);
        self.g[n + u] -= self.max_h * (p[1] - x_u[1]);
        self.lock_us.push(u as u32);
    }

    pub fn compute_step_size(&self) -> f32 {
        let n = self.n;
        let g = self.g.as_ptr();

        let mut num_v = f32x4_splat(0.);
        let n2 = 2 * n;
        let chunks = n2 / 4;
        unsafe {
            for ci in 0..chunks {
                let gv = v128_load(g.add(ci * 4) as *const v128);
                num_v = f32x4_add(num_v, f32x4_mul(gv, gv));
            }
        }
        let mut numerator = f32x4_extract_lane::<0>(num_v)
            + f32x4_extract_lane::<1>(num_v)
            + f32x4_extract_lane::<2>(num_v)
            + f32x4_extract_lane::<3>(num_v);
        for i in (chunks * 4)..n2 {
            numerator += self.g[i] * self.g[i];
        }

        let mut den = 0f32;
        unsafe {
            let gx = g;
            let gy = g.add(n);
            for p in &self.active {
                let (u, v) = (p.u as usize, p.v as usize);
                let dgx = *gx.add(u) - *gx.add(v);
                let dgy = *gy.add(u) - *gy.add(v);
                den += p.idk_x * dgx * dgx + p.idk_y * dgy * dgy;
            }
        }
        let mut denominator = -den;
        for &u in &self.lock_us {
            let u = u as usize;
            denominator += self.max_h * (self.g[u] * self.g[u] + self.g[n + u] * self.g[n + u]);
        }

        if denominator == 0. || !denominator.is_finite() {
            return 0.;
        }
        numerator / denominator
    }

    /// Locks are hard pins in the integrated tick path: rather than the original soft
    /// spring (which fights graph forces and oscillates visibly while dragging), pinned
    /// nodes are clamped to their lock position at every integrator stage.
    fn pin(&self, buf: *mut f32) {
        let n = self.n;
        for l in &self.locks {
            unsafe {
                *buf.add(l[0] as usize) = l[1];
                *buf.add(n + l[0] as usize) = l[2];
            }
        }
    }

    /// r = x0 - alpha * g
    fn descend_into(&mut self, x0: *const f32, r: *mut f32) {
        self.compute(x0 as *mut f32);
        let alpha = self.compute_step_size();
        let g = self.g.as_ptr();
        for i in 0..(2 * self.n) {
            unsafe {
                *r.add(i) = *x0.add(i) - alpha * *g.add(i);
            }
        }
        self.pin(r);
    }

    fn mid(x: *const f32, a: *const f32, out: *mut f32, n2: usize) {
        for i in 0..n2 {
            unsafe {
                let xi = *x.add(i);
                *out.add(i) = xi + (*a.add(i) - xi) / 2.0;
            }
        }
    }

    /// mode: 0 = RK4 (original webcola behavior), 1 = midpoint, 2 = plain gradient descent
    pub fn tick(&mut self, mode: u32) -> f32 {
        let n2 = 2 * self.n;
        self.track_huu = false;
        let x = self.x.as_mut_ptr();
        self.pin(x);
        let (a, b, c, d, ia, ib) = (
            self.a.as_mut_ptr(),
            self.b.as_mut_ptr(),
            self.c.as_mut_ptr(),
            self.d.as_mut_ptr(),
            self.ia.as_mut_ptr(),
            self.ib.as_mut_ptr(),
        );

        let mut disp = 0f32;
        match mode {
            0 => {
                self.descend_into(x, a);
                Self::mid(x, a, ia, n2);
                self.descend_into(ia, b);
                Self::mid(x, b, ib, n2);
                self.descend_into(ib, c);
                self.descend_into(c, d);
                for i in 0..n2 {
                    unsafe {
                        let nx = (*a.add(i) + 2.0 * *b.add(i) + 2.0 * *c.add(i) + *d.add(i)) / 6.0;
                        let dd = *x.add(i) - nx;
                        disp += dd * dd;
                        *x.add(i) = nx;
                    }
                }
            }
            1 => {
                self.descend_into(x, a);
                Self::mid(x, a, ia, n2);
                self.compute(ia);
                let alpha = self.compute_step_size();
                let g = self.g.as_ptr();
                for i in 0..n2 {
                    unsafe {
                        let nx = *x.add(i) - alpha * *g.add(i);
                        let dd = *x.add(i) - nx;
                        disp += dd * dd;
                        *x.add(i) = nx;
                    }
                }
                self.pin(x);
            }
            _ => {
                self.compute(x);
                let alpha = self.compute_step_size();
                let g = self.g.as_ptr();
                for i in 0..n2 {
                    unsafe {
                        let dd = alpha * *g.add(i);
                        disp += dd * dd;
                        *x.add(i) -= dd;
                    }
                }
                self.pin(x);
            }
        }
        disp
    }

    pub fn run(&mut self, iterations: u32, threshold: f32, mode: u32) -> f32 {
        let mut stress = f32::MAX;
        for _ in 0..iterations {
            let s = self.tick(mode);
            if (stress / s - 1.).abs() < threshold {
                return s;
            }
            stress = s;
        }
        stress
    }
}

#[wasm_bindgen]
pub fn fast_create(node_count: usize, d: Vec<f32>) -> *mut FastCtx {
    Box::into_raw(Box::new(FastCtx::new(d, node_count)))
}

#[wasm_bindgen]
pub fn fast_free(ctx: *mut FastCtx) {
    unsafe { drop(Box::from_raw(ctx)) };
}

#[wasm_bindgen]
pub fn fast_set_g(ctx: *mut FastCtx, g: Vec<f32>) {
    unsafe { (*ctx).set_g(g) };
}

#[wasm_bindgen]
pub fn fast_set_x(ctx: *mut FastCtx, x: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    ctx.x.copy_from_slice(&x);
}

#[wasm_bindgen]
pub fn fast_x_ptr(ctx: *mut FastCtx) -> *mut f32 {
    unsafe { (*ctx).x.as_mut_ptr() }
}

#[wasm_bindgen]
pub fn fast_g_ptr(ctx: *mut FastCtx) -> *mut f32 {
    unsafe { (*ctx).g.as_mut_ptr() }
}

#[wasm_bindgen]
pub fn fast_active_count(ctx: *mut FastCtx) -> usize {
    unsafe { (*ctx).active.len() }
}

#[wasm_bindgen]
pub fn fast_set_locks(ctx: *mut FastCtx, locks: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    ctx.locks.clear();
    for ch in locks.chunks_exact(3) {
        ctx.locks.push([ch[0], ch[1], ch[2]]);
    }
}

#[wasm_bindgen]
pub fn fast_tick(ctx: *mut FastCtx, mode: u32) -> f32 {
    unsafe { (*ctx).tick(mode) }
}

#[wasm_bindgen]
pub fn fast_run(ctx: *mut FastCtx, iterations: u32, threshold: f32, mode: u32) -> f32 {
    unsafe { (*ctx).run(iterations, threshold, mode) }
}

// Granular API mirroring the original engine, used for validation and A/B benchmarking
#[wasm_bindgen]
pub fn fast_compute(ctx: *mut FastCtx, mut x: Vec<f32>) -> Vec<f32> {
    let ctx = unsafe { &mut *ctx };
    ctx.track_huu = true;
    ctx.compute(x.as_mut_ptr());
    x
}

#[wasm_bindgen]
pub fn fast_step_size(ctx: *mut FastCtx) -> f32 {
    unsafe { (*ctx).compute_step_size() }
}

#[wasm_bindgen]
pub fn fast_apply_lock(ctx: *mut FastCtx, u: usize, p0: f32, p1: f32, x0u: f32, x1u: f32) {
    unsafe { (*ctx).apply_lock(u, [p0, p1], [x0u, x1u]) };
}
