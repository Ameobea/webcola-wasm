#![allow(non_snake_case)]

use rand::prelude::*;
use rand_pcg::Pcg32;
#[cfg(feature = "simd")]
use std::arch::wasm32::*;
use wasm_bindgen::prelude::*;

#[cfg(feature = "simd")]
mod fast2d;

pub struct Context<const DIMS: usize> {
    /// Number of nodes
    pub n: usize,
    /// Gradient vector
    pub g: Vec<f32>,
    /// Hessian matrix
    pub H: Vec<f32>,
    /// matrix of desired distances between pairs of nodes
    pub D: Vec<f32>,
    pub G: Vec<f32>,
    rng: Pcg32,
    min_d: f32,
    // snap_grid_size: f32,
    // snap_strength: f32,
    max_h: f32,
    /// Scratch buffer used during step size computation
    Hd: [Vec<f32>; DIMS],
    /// Scratch buffer used to hold distances between all nodes
    distances: Vec<f32>,
    /// Holds square root of the sums of all distances squared for all dimensions
    summed_distances: Vec<f32>,
    /// Holds flags indicating whether or not the inner condition passes or not for each pair of nodes
    inner_condition_flags: Vec<f32>,
}

/// This is a magical value that we expect no real distance to ever come out to.  It is used to represent
/// indices where u=v and so the distance is zero, but in order to facilitate efficient displacement checking
/// we must not have zero values in that array.
const U_EQ_V_PLACEHOLDER_VAL: f32 = 999_999_999_999_888_233.12128;

impl<const DIMS: usize> Context<DIMS> {
    pub fn new(D: Vec<f32>, G: Vec<f32>, node_count: usize) -> Self {
        let mut g: Vec<f32> = Vec::with_capacity(DIMS * node_count);
        let mut H: Vec<f32> = Vec::with_capacity(DIMS * node_count * node_count);
        let mut distances: Vec<f32> = Vec::with_capacity(DIMS * node_count * node_count);
        let inner_condition_flags: Vec<f32> = Vec::with_capacity(node_count * node_count);

        unsafe {
            g.set_len(DIMS * node_count);
            H.set_len(DIMS * node_count * node_count);
            distances.set_len(DIMS * node_count * node_count);
        };
        let mut Hd: [Vec<f32>; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for i in 0..DIMS {
            unsafe {
                Hd.as_mut_ptr().add(i).write(vec![0.; node_count]);
            }
        }

        let mut summed_distances: Vec<f32> = vec![0.; node_count * node_count];
        // This is an optimization to facilitate faster displacment checking.  We set values where u=v to non-zero values so that they aren't treated as needing displacement.
        for u in 0..node_count {
            for v in 0..node_count {
                if u != v {
                    continue;
                }

                unsafe {
                    *summed_distances.get_unchecked_mut(u * node_count + v) =
                        U_EQ_V_PLACEHOLDER_VAL;
                }
            }
        }

        let mut min_d = D
            .iter()
            .copied()
            .filter(|&x| !x.is_nan() && !x.is_infinite() && x > 0.)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f32::MAX);
        if min_d == f32::MAX {
            min_d = 1.;
        }

        Context {
            n: node_count,
            g,
            G,
            H,
            D,
            rng: Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7),
            min_d,
            // snap_grid_size: 100.,
            // snap_strength: 1000.,
            max_h: 0.,
            Hd,
            distances,
            summed_distances,
            inner_condition_flags,
        }
    }

    pub fn offset_dir(&mut self) -> [f32; DIMS] {
        let mut u: [f32; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut l = 0.;
        for i in 0..DIMS {
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

    fn get_H_mut(&mut self, i: usize, u: usize, v: usize) -> &mut f32 {
        let n = self.n;
        let dim_size = self.n * n;
        if cfg!(debug_assertions) {
            &mut self.H[dim_size * i + (u * n) + v]
        } else {
            unsafe { self.H.get_unchecked_mut(dim_size * i + (u * n) + v) }
        }
    }

    fn set_H(&mut self, i: usize, u: usize, v: usize, val: f32) {
        let n = self.n;
        let dim_size = self.n * n;
        if cfg!(debug_assertions) {
            self.H[dim_size * i + (u * n) + v] = val;
        } else {
            unsafe {
                *self.H.get_unchecked_mut(dim_size * i + (u * n) + v) = val;
            }
        }
    }

    /// Returns `true` if any displacements were applied
    #[inline(never)]
    fn apply_displacements(&mut self, x: &mut [f32]) -> bool {
        let n = self.n;
        let mut did_apply = false;

        for u in 0..n {
            for v in 0..n {
                if u == v {
                    continue;
                }

                // We have to re-compute distances here since we clobber the `summed_distances` array
                // if the flags match to skip the inner work loop
                let mut summed_distance_squared = 0.;
                for i in 0..DIMS {
                    let dist = unsafe { *self.distances.get_unchecked((i * n * n) + (u * n) + v) };
                    summed_distance_squared += dist * dist;
                }
                let summed_distance = summed_distance_squared.sqrt();

                if summed_distance > 0.000000001 {
                    continue;
                }

                did_apply = true;

                let rd = self.offset_dir();

                for i in 0..DIMS {
                    if cfg!(debug_assertions) {
                        x[i * n + v] += rd[i];
                    } else {
                        unsafe {
                            *x.get_unchecked_mut(i * n + v) += rd[i];
                        }
                    }
                }
            }
        }

        did_apply
    }

    #[cfg(not(feature = "simd"))]
    fn compute_distances(&mut self, x: &mut [f32]) -> bool {
        let n = self.n;
        let mut needs_displace = false;

        for i in 0..DIMS {
            let mut ix = 0;

            for u in 0..n {
                for v in 0..n {
                    unsafe {
                        let out_ix = (i * n * n) + (u * n) + v;

                        let distance = *x.get_unchecked(i * n + u) - *x.get_unchecked(i * n + v);
                        let distance_squared = distance * distance;
                        *self.distances.get_unchecked_mut(out_ix) = distance;
                        *self.summed_distances.get_unchecked_mut(ix) += distance_squared;
                        if i == DIMS - 1 {
                            let sqrtd = self.summed_distances.get_unchecked_mut(ix).sqrt();
                            *self.summed_distances.get_unchecked_mut(ix) = sqrtd;

                            if sqrtd < 0.000000001 {
                                needs_displace = true;
                            }

                            // compute condition flags that are used in the inner loop.  We can do it here
                            // using SIMD and store the flags in memory to be read out later.
                            //
                            // The gist of what we're computing is this:
                            // (sqrted > ideal_distance) && (weight > 1.)
                            let weight = *self.G.get_unchecked(ix);
                            let ideal_distance = *self.D.get_unchecked(ix);
                            let flag = (sqrtd > ideal_distance) && (weight > 1.);
                            *(self.inner_condition_flags.get_unchecked_mut(ix) as *mut f32
                                as *mut i32) = if flag { -1i32 } else { 0i32 };
                        }
                    }

                    ix += 1;
                }
            }
        }

        needs_displace
    }

    // #[inline(never)]
    #[cfg(feature = "simd")]
    fn compute_distances(&mut self, x: &mut [f32]) -> bool {
        let n = self.n;
        let chunk_count = (n - (n % 4)) / 4;

        // This is a set of flags to facilitate efficient SIMD.  If any of the contained elements are non-zero, then displacements are needed
        let mut needs_displace = false;
        // all 1s
        let mut all_over_displacement_threshold =
            unsafe { f32x4_splat(std::mem::transmute(-1i32)) };
        let displacement_threshold = unsafe { f32x4_splat(0.000000001) };

        for i in 0..DIMS {
            for u in 0..n {
                unsafe {
                    let u_vector =
                        v128_load32_splat(x.get_unchecked(i * n + u) as *const f32 as *const u32);

                    for v_chunk_ix in 0..chunk_count {
                        let v_vector =
                            v128_load(x.as_ptr().add(i * n + v_chunk_ix * 4) as *const v128);

                        let distances = f32x4_sub(u_vector, v_vector);
                        let out_ix = (i * n * n) + (u * n) + v_chunk_ix * 4;

                        let distances_out_ptr =
                            self.distances.as_mut_ptr().add(out_ix) as *mut v128;
                        v128_store(distances_out_ptr, distances);

                        let distances_squared = f32x4_mul(distances, distances);

                        let summed_distances_squared_ptr = self
                            .summed_distances
                            .as_mut_ptr()
                            .add(u * n + v_chunk_ix * 4)
                            as *mut v128;
                        let summed_distances_squared_v_orig =
                            v128_load(summed_distances_squared_ptr);
                        let summed_distances_squared_v =
                            f32x4_add(distances_squared, summed_distances_squared_v_orig);

                        // sqrt it on the last iteration
                        if i == DIMS - 1 {
                            let sqrted = f32x4_sqrt(summed_distances_squared_v);

                            // check here if we need to apply displacements
                            let over_displacement_threshold =
                                f32x4_gt(sqrted, displacement_threshold);
                            all_over_displacement_threshold = v128_and(
                                all_over_displacement_threshold,
                                over_displacement_threshold,
                            );

                            // compute condition flags that are used in the inner loop.  We can do it here
                            // using SIMD and store the flags in memory to be read out later.
                            //
                            // The gist of what we're computing is this:
                            // (sqrted > ideal_distance) && (weight > 1.)
                            let ideal_distances_v = v128_load(
                                self.D.get_unchecked(u * n + v_chunk_ix * 4) as *const f32
                                    as *const _,
                            );
                            let weights_v = v128_load(self.G.get_unchecked(u * n + v_chunk_ix * 4)
                                as *const f32
                                as *const _);

                            let flags = v128_and(
                                f32x4_gt(sqrted, ideal_distances_v),
                                f32x4_gt(weights_v, f32x4_splat(1.)),
                            );
                            v128_store(
                                self.inner_condition_flags
                                    .get_unchecked_mut(u * n + v_chunk_ix * 4)
                                    as *mut f32 as *mut _,
                                flags,
                            );

                            let zeroed_by_flags = v128_andnot(sqrted, flags);

                            // We want to leave sums where u=v as they are in or
                            let eq_placeholder = f32x4_eq(
                                summed_distances_squared_v_orig,
                                f32x4_splat(U_EQ_V_PLACEHOLDER_VAL),
                            );
                            // we pre-zero summed distances if the flag is zero, but we need to preserve elements
                            // where u=v to prevent them from getting zeroed out
                            v128_store(
                                summed_distances_squared_ptr,
                                v128_bitselect(
                                    summed_distances_squared_v_orig,
                                    zeroed_by_flags,
                                    eq_placeholder,
                                ),
                            );
                        } else {
                            v128_store(summed_distances_squared_ptr, summed_distances_squared_v);
                        }
                    }

                    // Multiply the last partial chunk manually
                    for v in (chunk_count * 4)..n {
                        if u == v {
                            continue;
                        }

                        let out_ix = (i * n * n) + (u * n) + v;

                        if i == 0 {
                            *self.summed_distances.get_unchecked_mut(u * n + v) = 0.;
                        }

                        let distance = *x.get_unchecked(i * n + u) - *x.get_unchecked(i * n + v);
                        let distance_squared = distance * distance;
                        *self.distances.get_unchecked_mut(out_ix) = distance;
                        *self.summed_distances.get_unchecked_mut(u * n + v) += distance_squared;
                        if i == DIMS - 1 {
                            let sqrtd = self.summed_distances.get_unchecked_mut(u * n + v).sqrt();
                            *self.summed_distances.get_unchecked_mut(u * n + v) = sqrtd;

                            if sqrtd < 0.000000001 {
                                needs_displace = true;
                            }

                            // compute condition flags that are used in the inner loop.  We can do it here
                            // using SIMD and store the flags in memory to be read out later.
                            //
                            // The gist of what we're computing is this:
                            // (sqrted > ideal_distance) && (weight > 1.)
                            let weight = *self.G.get_unchecked(u * n + v);
                            let ideal_distance = *self.D.get_unchecked(u * n + v);
                            let flag = (sqrtd > ideal_distance) && (weight > 1.);
                            *(self.inner_condition_flags.get_unchecked_mut(u * n + v) as *mut f32
                                as *mut i32) = if flag { -1i32 } else { 0i32 };
                        }
                    }
                }
            }
        }

        unsafe { needs_displace || !i32x4_all_true(all_over_displacement_threshold) }
    }

    pub fn compute(&mut self, x: &mut [f32]) {
        // Pre-compute distances and distances squared between all nodes
        loop {
            let needs_displacement = self.compute_distances(x);
            if !needs_displacement {
                break;
            }

            let did_apply = self.apply_displacements(x);
            if !did_apply {
                break;
            }
        }

        let mut max_h: f32 = 0.;
        let n = self.n;

        self.g.fill(0.);

        // across all nodes u
        let mut ix = 0;
        for u in 0..self.n {
            // Hessian diagonal
            let mut Huu: [f32; DIMS] = [0.; DIMS];

            // across all nodes v
            for v in 0..self.n {
                if u == v {
                    ix += 1;
                    continue;
                }

                // The original comment from the JS for what roughly equates to this bit of code:
                //
                // "ignore long range attractions for nodes not immediately connected (P-stress)"
                //
                // This value is pre-computed in `compute_distances` and corresponds to some logic that
                // boils down to:
                //
                // (sqrt(summed_distances_squared_per_dimension) > ideal_distance) && (weight > 1.)
                //
                // That in turn replaces this logic:
                //
                // (distance > ideal_distance && weight > 1.) || !ideal_distance.is_finite()
                //
                // We've done some hacky stuff in the JS wrapper where `ideal_distance` is set to a hugely
                // negative number and `weight` is set to 1000. if `ideal_distance` is equal to infinity.
                // This removes the need to do the expensive `is_finite` check but diverges greatly from
                // what the original JS code did.
                //
                // I don't think it's possible in the current codebase for the buffers where `ideal_distance`
                // and `weight` come from to be messed with/read after the first run, but if that did happen
                // it would almost certainly break this code.
                let flag = unsafe {
                    *(self.inner_condition_flags.get_unchecked(ix) as *const f32 as *const u32)
                };
                // if distance > ideal_distance && weight > 1. {
                if flag != 0 {
                    for i in 0..DIMS {
                        self.set_H(i, u, v, 0.);
                    }
                    ix += 1;
                    continue;
                }

                let distance = if cfg!(debug_assertions) {
                    let val = self.summed_distances[ix];
                    self.summed_distances[ix] = 0.;
                    val
                } else {
                    unsafe {
                        let ptr = self.summed_distances.get_unchecked_mut(ix);
                        let val = *ptr;
                        *ptr = 0.;
                        val
                    }
                };

                let ideal_distance = if cfg!(debug_assertions) {
                    self.D[ix]
                } else {
                    unsafe { *self.D.get_unchecked(ix) }
                };

                if cfg!(debug_assertions) {
                    if ideal_distance == 0. {
                        panic!("ideal_distance={}; u={}, v={}", ideal_distance, u, v);
                    }
                }

                // \/ This is no longer necesary since we assume `weight` is always 1, see the comment block above
                //
                // Comment from the original JS:
                //
                // weight > 1 was just an indicator - this is an arcane interface,
                // but we are trying to be economical storing and passing node pair info
                // if weight > 1. {
                //     weight = 1.;
                // }

                // We do not support group/dummy nodes, and so we are able to simplify a lot
                // of the math here that treats weight as a free variable
                let weight = 1.;

                let distance_squared = distance * distance;
                let ideal_distance_squared = ideal_distance * ideal_distance;
                let gs =
                    2. * weight * (distance - ideal_distance) / (ideal_distance_squared * distance);
                let distance_cubed = distance_squared * distance;
                let hs = 2. * -weight / (ideal_distance_squared * distance_cubed);

                if cfg!(debug_assertions) {
                    if !gs.is_finite() {
                        if !distance_squared.is_normal() {
                            panic!(
                                "bad distance squared: {}, u={}, v={}",
                                distance_squared, u, v
                            );
                        } else if !weight.is_finite() {
                            panic!("bad weight: {}", weight);
                        } else if !distance.is_normal() {
                            panic!();
                        } else if !ideal_distance.is_normal() {
                            panic!();
                        }

                        panic!();
                    }
                }

                for i in 0..DIMS {
                    let distance = if cfg!(debug_assertions) {
                        self.distances[(i * n * n) + ix]
                    } else {
                        unsafe { *self.distances.get_unchecked((i * n * n) + ix) }
                    };

                    if cfg!(debug_assertions) {
                        self.g[i * n + u] += distance * gs;
                    } else {
                        unsafe { *self.g.get_unchecked_mut(i * n + u) += distance * gs };
                    }
                    let idk = hs
                        * (2. * distance_cubed
                            + ideal_distance * (distance * distance - distance_squared));
                    self.set_H(i, u, v, idk);
                    Huu[i] -= idk;
                }

                ix += 1;
            }

            for i in 0..DIMS {
                self.set_H(i, u, u, Huu[i]);
                if Huu[i] > max_h {
                    max_h = Huu[i];
                }
            }
        }

        // Grid snap forces: TODO

        // used by locks
        self.max_h = max_h;
    }

    pub fn apply_lock(&mut self, u: usize, p: [f32; DIMS], x_i_u: [f32; DIMS]) {
        let n = self.n;
        for i in 0..DIMS {
            *self.get_H_mut(i, u, u) += self.max_h;
            if cfg!(debug_assertions) {
                self.g[i * n + u] -= self.max_h * (p[i] - x_i_u[i]);
            } else {
                unsafe {
                    *self.g.get_unchecked_mut(i * n + u) -= self.max_h * (p[i] - x_i_u[i]);
                }
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    fn dot_prod(a: *const f32, b: *const f32, count: u64) -> f32 {
        let mut out = 0.;
        for i in 0..count {
            out += unsafe { *a.add(i as usize) * *b.add(i as usize) };
        }
        out
    }

    #[cfg(feature = "simd")]
    fn dot_prod(a: *const f32, b: *const f32, count: u64) -> f32 {
        let mut vector_sum = unsafe { f32x4_splat(0.) };
        const CHUNK_SIZE: u64 = 4 * 4;
        let chunk_count = (count - (count % CHUNK_SIZE)) / CHUNK_SIZE;

        let mut i = 0u64;
        let max_i = chunk_count * CHUNK_SIZE;
        while i != max_i {
            unsafe {
                let a_n = v128_load(a.add(i as usize) as *const v128);
                let b_n = v128_load(b.add(i as usize) as *const v128);
                let multiplied = f32x4_mul(a_n, b_n);
                vector_sum = f32x4_add(vector_sum, multiplied);

                let a_n = v128_load(a.add(i as usize + 4) as *const v128);
                let b_n = v128_load(b.add(i as usize + 4) as *const v128);
                let multiplied = f32x4_mul(a_n, b_n);
                vector_sum = f32x4_add(vector_sum, multiplied);

                let a_n = v128_load(a.add(i as usize + 8) as *const v128);
                let b_n = v128_load(b.add(i as usize + 8) as *const v128);
                let multiplied = f32x4_mul(a_n, b_n);
                vector_sum = f32x4_add(vector_sum, multiplied);

                let a_n = v128_load(a.add(i as usize + 12) as *const v128);
                let b_n = v128_load(b.add(i as usize + 12) as *const v128);
                let multiplied = f32x4_mul(a_n, b_n);
                vector_sum = f32x4_add(vector_sum, multiplied);
            }

            i += CHUNK_SIZE;
        }

        let mut sum = unsafe {
            f32x4_extract_lane::<0>(vector_sum)
                + f32x4_extract_lane::<1>(vector_sum)
                + f32x4_extract_lane::<2>(vector_sum)
                + f32x4_extract_lane::<3>(vector_sum)
        };

        // Remainder
        for i in max_i..(count as u64) {
            sum += unsafe { *a.add(i as usize) * *b.add(i as usize) };
        }

        sum
    }

    /// result r = matrix m * vector v
    // #[inline(never)]
    fn right_multiply<'a>(
        m: *const f32,
        m_chunk_count: usize,
        m_chunk_size: u64,
        v: *const f32,
        r: &mut [f32],
    ) {
        for i in 0..m_chunk_count {
            let mn = unsafe { m.add(i * m_chunk_size as usize) };
            if cfg!(debug_assertions) {
                r[i] = Self::dot_prod(mn, v, m_chunk_size);
            } else {
                unsafe { *r.get_unchecked_mut(i) = Self::dot_prod(mn, v, m_chunk_size) };
            }
        }
    }

    /// Computes the optimal step size to take in direction d using the derivative information in this.g and this.H
    /// returns the scalar multiplier to apply to d to get the optimal step
    ///
    /// Only computes step size for `this.g`; computing step size with `this.e` is not implemented.
    pub fn compute_step_size(&mut self) -> f32 {
        let mut numerator = 0.;
        let mut denominator = 0.;
        let H_dim_size = self.n * self.n;
        let n = self.n;
        let n_u64 = self.n as u64;

        for i in 0..DIMS {
            let gn = unsafe { self.g.as_ptr().add(i * n) };

            numerator += Self::dot_prod(gn, gn, n_u64);
            let Hd_i = if cfg!(debug_assertions) {
                &mut self.Hd[i]
            } else {
                unsafe { self.Hd.get_unchecked_mut(i) }
            };

            Self::right_multiply(
                if cfg!(debug_assertions) {
                    &self.H[(i * H_dim_size)..(i * H_dim_size + H_dim_size)]
                } else {
                    unsafe {
                        self.H
                            .get_unchecked((i * H_dim_size)..(i * H_dim_size + H_dim_size))
                    }
                }
                .as_ptr(),
                n,
                n_u64,
                gn,
                Hd_i,
            );

            denominator += Self::dot_prod(gn, Hd_i.as_ptr(), n_u64);
        }

        if denominator == 0. || !denominator.is_finite() {
            return 0.;
        }
        return 1. * numerator / denominator;
    }
}

#[wasm_bindgen]
pub fn create_derivative_computer_ctx_2d(
    node_count: usize,
    D: Vec<f32>,
    G: Vec<f32>,
) -> *mut Context<2> {
    if cfg!(debug_assertions) {
        console_error_panic_hook::set_once();
    }

    debug_assert_eq!(D.len(), node_count * node_count);
    if !G.is_empty() {
        debug_assert_eq!(G.len(), node_count * node_count);
    }

    let ctx: Context<2> = Context::new(
        D,
        if G.is_empty() {
            vec![1.; node_count * node_count]
        } else {
            G
        },
        node_count,
    );
    Box::into_raw(Box::new(ctx))
}

#[wasm_bindgen]
pub fn create_derivative_computer_ctx_3d(
    node_count: usize,
    D: Vec<f32>,
    G: Vec<f32>,
) -> *mut Context<3> {
    if cfg!(debug_assertions) {
        console_error_panic_hook::set_once();
    }

    debug_assert_eq!(D.len(), node_count * node_count);
    if !G.is_empty() {
        debug_assert_eq!(G.len(), node_count * node_count);
    }

    let ctx: Context<3> = Context::new(
        D,
        if G.is_empty() {
            vec![1.; node_count * node_count]
        } else {
            G
        },
        node_count,
    );
    Box::into_raw(Box::new(ctx))
}

#[wasm_bindgen]
pub fn compute_2d(ctx_ptr: *mut Context<2>, mut x: Vec<f32>) -> Vec<f32> {
    let ctx: &mut Context<2> = unsafe { &mut *ctx_ptr };
    ctx.compute(&mut x);
    x
}

#[wasm_bindgen]
pub fn compute_3d(ctx_ptr: *mut Context<3>, mut x: Vec<f32>) -> Vec<f32> {
    let ctx: &mut Context<3> = unsafe { &mut *ctx_ptr };
    ctx.compute(&mut x);
    x
}

#[wasm_bindgen]
pub fn apply_lock_2d(
    ctx_ptr: *mut Context<2>,
    u: usize,
    p_0: f32,
    p_1: f32,
    x_0_u: f32,
    x_1_u: f32,
) {
    let ctx = unsafe { &mut *ctx_ptr };
    ctx.apply_lock(u, [p_0, p_1], [x_0_u, x_1_u]);
}

#[wasm_bindgen]
pub fn apply_lock_3d(
    ctx_ptr: *mut Context<3>,
    u: usize,
    p_0: f32,
    p_1: f32,
    p_2: f32,
    x_0_u: f32,
    x_1_u: f32,
    x_2_u: f32,
) {
    let ctx = unsafe { &mut *ctx_ptr };
    ctx.apply_lock(u, [p_0, p_1, p_2], [x_0_u, x_1_u, x_2_u]);
}

#[wasm_bindgen]
pub fn compute_step_size_2d(ctx_ptr: *mut Context<2>) -> f32 {
    let ctx = unsafe { &mut *ctx_ptr };
    ctx.compute_step_size()
}

#[wasm_bindgen]
pub fn compute_step_size_3d(ctx_ptr: *mut Context<3>) -> f32 {
    let ctx = unsafe { &mut *ctx_ptr };
    ctx.compute_step_size()
}

#[wasm_bindgen]
pub fn get_memory() -> JsValue {
    wasm_bindgen::memory()
}

////////////////////////////////////////////////////////////////

// D Getters
#[wasm_bindgen]
pub fn get_D_2d(ctx: *mut Context<2>) -> *mut f32 {
    let ctx = unsafe { &mut *ctx };
    ctx.D.as_mut_ptr()
}

#[wasm_bindgen]
pub fn get_D_3d(ctx: *mut Context<3>) -> *mut f32 {
    let ctx = unsafe { &mut *ctx };
    ctx.D.as_mut_ptr()
}

// g Getters
#[wasm_bindgen]
pub fn get_g_2d(ctx: *mut Context<2>) -> *mut f32 {
    unsafe { (*ctx).g.as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_g_3d(ctx: *mut Context<3>) -> *mut f32 {
    unsafe { (*ctx).g.as_mut_ptr() }
}

#[wasm_bindgen]
pub fn set_G_2d(ctx: *mut Context<2>, new_G: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    // assert_eq!(new_G.len(), ctx.n * ctx.n);
    ctx.G = if new_G.is_empty() {
        vec![1.; ctx.n * ctx.n]
    } else {
        new_G
    };
}

#[wasm_bindgen]
pub fn set_G_3d(ctx: *mut Context<3>, new_G: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    ctx.G = if new_G.is_empty() {
        vec![1.; ctx.n * ctx.n]
    } else {
        new_G
    };
}
