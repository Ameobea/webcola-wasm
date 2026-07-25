import type { DerivativeComputerWasmInst } from "wasmEngine";

    /**
     * Descent respects a collection of locks over nodes that should not move
     * @class Locks
     */
    export class Locks {
        locks: { [key:number]:number[]} = {};
        /**
         * add a lock on the node at index id
         * @method add
         * @param id index of node to be locked
         * @param x required position for node
         */
        add(id: number, x: number[]) {
/* DEBUG
            if (isNaN(x[0]) || isNaN(x[1])) debugger;
DEBUG */
            this.locks[id] = x;
        }
        /**
         * @method clear clear all locks
         */
        clear() {
            this.locks = {};
        }
        /**
         * @isEmpty
         * @returns false if no locks exist
         */
        isEmpty(): boolean {
            for (var l in this.locks) return false;
            return true;
        }
        /**
         * perform an operation on each lock
         * @apply
         */
        apply(f: (id: number, x: number[]) => void) {
            for (var l in this.locks) {
                f(Number(l), this.locks[l]);
            }
        }
    }

    const BYTES_PER_F32 = 32 / 8;

    /**
     * Uses a gradient descent approach to reduce a stress or p-stress goal function over a graph with specified ideal edge lengths or a square matrix of dissimilarities.
     * The standard stress function over a graph nodes with position vectors x,y,z is (mathematica input):
     *   stress[x_,y_,z_,D_,w_]:=Sum[w[[i,j]] (length[x[[i]],y[[i]],z[[i]],x[[j]],y[[j]],z[[j]]]-d[[i,j]])^2,{i,Length[x]-1},{j,i+1,Length[x]}]
     * where: D is a square matrix of ideal separations between nodes, w is matrix of weights for those separations
     *        length[x1_, y1_, z1_, x2_, y2_, z2_] = Sqrt[(x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2]
     * below, we use wij = 1/(Dij^2)
     *
     * @class Descent
     */
    export class Descent {
        private wasm: DerivativeComputerWasmInst;
        private ctxPtr: number;
        /**
         * `true` when the specialized 2D engine (fast_* exports) is in use.  Positions then
         * live in wasm memory and `x` is a pair of views into it; the whole integrator runs
         * wasm-side per `rungeKutta` call.
         */
        private fast: boolean = false;
        private allD: Float32Array | null = null;
        private xViews: Float32Array[] | null = null;
        private lastLockCount = 0;

        /** integrator used by the fast engine: 0 = RK4 (original), 1 = midpoint, 2 = gradient descent */
        public integratorMode = 0;
        /** number of integrator steps per rungeKutta() call (fast engine only) */
        public substeps = 1;

        public threshold: number = 0.0001;
        /** gradient vector
         * @property g {Float32Array[]}
         */
        public get g(): Float32Array[] {
            const memory: WebAssembly.Memory = this.wasm.get_memory();
            const memoryView = new Float32Array(memory.buffer);

            const gPtr = this.fast
                ? (this.wasm as any).fast_g_ptr(this.ctxPtr)
                : this.k === 2 ? this.wasm.get_g_2d(this.ctxPtr) : this.wasm.get_g_3d(this.ctxPtr);
            const gOffset = gPtr / BYTES_PER_F32;
            return new Array(this.k)
                .fill(null)
                .map((_, i) => memoryView.subarray(gOffset + i * this.n, gOffset + i * this.n + this.n));
        }
        public set G(newG: Float32Array[] | null)  {
            const allG = (() => {
                if (newG) {
                    const allG = new Float32Array(this.n * this.n);
                    newG.forEach((Gn, i) => allG.set(Gn, i * this.n));
                    return allG;
                } else {
                    return new Float32Array();
                }
            })();

            if (this.fast) {
                (this.wasm as any).fast_set_g(this.ctxPtr, allG);
            } else if (this.k === 2) {
                this.wasm.set_G_2d(this.ctxPtr, allG);
            } else if (this.k === 3) {
                this.wasm.set_G_3d(this.ctxPtr, allG);
            } else {
                throw new Error('Invalid dimensionality');
            }
        }
       /** positions vector
         * @property x {number[][]}
         */
        private _x: Float32Array[];
        public get x(): Float32Array[] {
            if (!this.fast) {
                return this._x;
            }
            // views into wasm memory; rebuilt if the memory grew (which detaches old buffers)
            if (!this.xViews || this.xViews[0].length === 0) {
                const memView = new Float32Array((this.wasm.get_memory() as WebAssembly.Memory).buffer);
                const off = (this.wasm as any).fast_x_ptr(this.ctxPtr) / BYTES_PER_F32;
                this.xViews = [
                    memView.subarray(off, off + this.n),
                    memView.subarray(off + this.n, off + 2 * this.n),
                ];
            }
            return this.xViews;
        }
        public set x(v: Float32Array[]) {
            this._x = v;
        }
        /**
         * @property k {number} dimensionality
         */
        public k: number;
        /**
         * number of data-points / nodes / size of vectors/matrices
         * @property n {number}
         */
        public n: number;
        /**
         * matrix of desired distances between pairs of nodes
         */
         public get D(): Float32Array[] {
            if (this.fast) {
                return new Array(this.n)
                    .fill(null)
                    .map((_, i) => this.allD!.subarray(i * this.n, i * this.n + this.n));
            }

            const memory: WebAssembly.Memory = this.wasm.get_memory();
            const memoryView = new Float32Array(memory.buffer);

            const DPtr = this.k === 2 ? this.wasm.get_D_2d(this.ctxPtr) : this.wasm.get_D_3d(this.ctxPtr);
            const DOffset = DPtr / BYTES_PER_F32;
            return new Array(this.n)
                .fill(null)
                .map((_, i) => memoryView.subarray(DOffset + i * this.n, DOffset + i * this.n + this.n));
        }

        public computeDerivatives(x: Float32Array[]) {
            if (this.fast) {
                const packed = new Float32Array(this.n * 2);
                packed.set(x[0], 0);
                packed.set(x[1], this.n);
                const outX: Float32Array = (this.wasm as any).fast_compute(this.ctxPtr, packed);
                x[0].set(outX.subarray(0, this.n));
                x[1].set(outX.subarray(this.n, 2 * this.n));

                if (!this.locks.isEmpty()) {
                    this.locks.apply((u, p) => {
                        (this.wasm as any).fast_apply_lock(this.ctxPtr, u, p[0], p[1], x[0][u], x[1][u]);
                    });
                }
                return;
            }

            if (this.k === 2) {
                const packedX = (() => {
                    const packed = new Float32Array(x[0].length * this.k);
                    x.forEach((xn, i) => packed.set(xn, i * this.n));
                    return packed;
                })();
                const outX = this.wasm.compute_2d(this.ctxPtr, packedX);

                if (x) {
                    x.forEach((xn, i) => {
                        const slice = outX.subarray(i * this.n, i * this.n + this.n);
                        xn.set(slice);
                    })
                }
            } else if (this.k === 3) {
                const packedX = (() => {
                    const packed = new Float32Array(x[0].length * this.k);
                    x.forEach((xn, i) => packed.set(xn, i * this.n));
                    return packed;
                })();
                const outX = this.wasm.compute_3d(this.ctxPtr, packedX);

                if (x) {
                    x.forEach((xn, i) => {
                        const slice = outX.subarray(i * this.n, i * this.n + this.n);
                        xn.set(slice);
                    })
                }
            } else {
                throw new Error('Invalid dimensionality');
            }

            if (!this.locks.isEmpty()) {
                this.locks.apply((u, p) => {
                    if (this.k === 2) {
                        this.wasm.apply_lock_2d(this.ctxPtr, u, p[0], p[1], x[0][u], x[1][u]);
                    } else if (this.k === 3) {
                        this.wasm.apply_lock_3d(this.ctxPtr, u, p[0], p[1], p[2], x[0][u], x[1][u], x[2][u]);
                    } else {
                        throw new Error('Invalid dimensionality');
                    }
                });
            }
        }

        public locks: Locks;

        private static zeroDistance: number = 1e-10;
        private minD: number;

        // pool of arrays of size n used internally, allocated in constructor
        private a: Float32Array[];
        private b: Float32Array[];
        private c: Float32Array[];
        private d: Float32Array[];
        private e: Float32Array[];
        private ia: Float32Array[];
        private ib: Float32Array[];
        private xtmp: number[][];


        // Parameters for grid snap stress.
        // TODO: Make a pluggable "StressTerm" class instead of this
        // mess.
        public numGridSnapNodes: number = 0;
        public snapGridSize: number = 100;
        public snapStrength: number = 1000;
        public scaleSnapByMaxH: boolean = false;

        private random = new PseudoRandom();

        public project: { (x0: Float32Array, y0: Float32Array, r: Float32Array): void }[] = null;

        private setupWasm(D: number[][], G: number[][] | null = null) {
            const allD = new Float32Array(this.n * this.n);
            const allG = G ? new Float32Array(this.n * this.k) : new Float32Array(0);
            D.forEach((dn, i) => {
                allD.set(dn, i * this.n);
            });
            if (G) {
                G.forEach((gn, i) => {
                    allG.set(gn, i * this.n);
                });
            }

            allD.forEach((d, i) => {
                if (d === Infinity) {
                    allD[i] = -10000000; // ideal distance
                    allG[i] = 1000.; // weight
                }
            });

            if (this.fast) {
                this.allD = allD;
                this.ctxPtr = (this.wasm as any).fast_create(this.n, allD);
                if (G) {
                    (this.wasm as any).fast_set_g(this.ctxPtr, allG);
                }
                return;
            }

            const createrFn = this.k === 2 ? this.wasm.create_derivative_computer_ctx_2d : this.wasm.create_derivative_computer_ctx_3d;
            this.ctxPtr = createrFn(this.n, allD, allG);
        }

        /**
         * @method constructor
         * @param x {number[][]} initial coordinates for nodes
         * @param D {number[][]} matrix of desired distances between pairs of nodes
         * @param G {number[][]} [default=null] if specified, G is a matrix of weights for goal terms between pairs of nodes.
         * If G[i][j] > 1 and the separation between nodes i and j is greater than their ideal distance, then there is no contribution for this pair to the goal
         * If G[i][j] <= 1 then it is used as a weighting on the contribution of the variance between ideal and actual separation between i and j to the goal function
         */
        constructor(x: number[][], D: number[][], G: number[][] = null, wasm: DerivativeComputerWasmInst) {
            this.wasm = wasm;
            this.k = x.length; // dimensionality
            var n = this.n = x[0].length; // number of nodes
            this.fast = this.k === 2 && typeof (wasm as any).fast_create === 'function';

            // Set up Wasm context
            this.setupWasm(D, G);

            if (this.fast) {
                const packed = new Float32Array(2 * n);
                packed.set(x[0], 0);
                packed.set(x[1], n);
                (this.wasm as any).fast_set_x(this.ctxPtr, packed);
            } else {
                this.x = x.map(xn => new Float32Array(xn));
            }

            this.a = new Array(this.k);
            this.b = new Array(this.k);
            this.c = new Array(this.k);
            this.d = new Array(this.k);
            this.e = new Array(this.k);
            this.ia = new Array(this.k);
            this.ib = new Array(this.k);
            this.xtmp = new Array(this.k);
            this.locks = new Locks();
            if (this.fast) {
                this.minD = 1;
                return;
            }
            this.minD = Number.MAX_VALUE;
            var i = n, j;
            while (i--) {
                j = n;
                while (--j > i) {
                    var d = D[i][j];
                    if (d > 0 && d < this.minD) {
                        this.minD = d;
                    }
                }
            }
            if (this.minD === Number.MAX_VALUE) this.minD = 1;
            i = this.k;
            while (i--) {
                j = n;
                this.a[i] = new Float32Array(n);
                this.b[i] = new Float32Array(n);
                this.c[i] = new Float32Array(n);
                this.d[i] = new Float32Array(n);
                this.e[i] = new Float32Array(n);
                this.ia[i] = new Float32Array(n);
                this.ib[i] = new Float32Array(n);
                this.xtmp[i] = new Array(n);
            }
        }

        public static createSquareMatrix(n: number, f: (i: number, j: number) => number): number[][] {
            var M = new Array(n);
            for (var i = 0; i < n; ++i) {
                M[i] = new Array(n);
                for (var j = 0; j < n; ++j) {
                    M[i][j] = f(i, j);
                }
            }
            return M;
        }

        private offsetDir(): number[] {
            var u = new Array(this.k);
            var l = 0;
            for (var i = 0; i < this.k; ++i) {
                var x = u[i] = this.random.getNextBetween(0.01, 1) - 0.5;
                l += x * x;
            }
            l = Math.sqrt(l);
            return u.map(x=> x *= this.minD / l);
        }

        private static dotProd(a: Float32Array, b: Float32Array): number {
            var x = 0, i = a.length;
            while (i--) x += a[i] * b[i];
            return x;
        }

        // result r = matrix m * vector v
        private static rightMultiply(m: Float32Array[], v: Float32Array, r: Float32Array) {
            var i = m.length;
            while (i--) r[i] = Descent.dotProd(m[i], v);
        }

        // computes the optimal step size to take in direction d using the
        // derivative information in this.g and this.H
        // returns the scalar multiplier to apply to d to get the optimal step
        public computeStepSize(): number {
            if (this.fast) {
                return (this.wasm as any).fast_step_size(this.ctxPtr);
            }
            if (this.k === 2) {
                return this.wasm.compute_step_size_2d(this.ctxPtr);
            } else if (this.k === 3) {
                return this.wasm.compute_step_size_3d(this.ctxPtr);
            } else {
                throw new Error('Invalid dimensionality');
            }
        }

        public reduceStress(): number {
            this.computeDerivatives(this.x);
            var alpha = this.computeStepSize();
            const thisG = this.g;
            for (var i = 0; i < this.k; ++i) {
                this.takeDescentStep(this.x[i], thisG[i], alpha);
            }
            return this.computeStress();
        }

        private static copy(a: Float32Array[], b: Float32Array[]): void {
            var m = a.length, n = b[0].length;
            for (var i = 0; i < m; ++i) {
                for (var j = 0; j < n; ++j) {
                    b[i][j] = a[i][j];
                }
            }
        }

        // takes a step of stepSize * d from x0, and then project against any constraints.
        // result is returned in r.
        // x0: starting positions
        // r: result positions will be returned here
        // d: unconstrained descent vector
        // stepSize: amount to step along d
        private stepAndProject(x0: Float32Array[], r: Float32Array[], d: Float32Array[], stepSize: number): void {
            Descent.copy(x0, r);
            this.takeDescentStep(r[0], d[0], stepSize);
            if (this.project) this.project[0](x0[0], x0[1], r[0]);
            this.takeDescentStep(r[1], d[1], stepSize);
            if (this.project) this.project[1](r[0], x0[1], r[1]);

            // todo: allow projection against constraints in higher dimensions
            for (var i = 2; i < this.k; i++)
                this.takeDescentStep(r[i], d[i], stepSize);

            // the following makes locks extra sticky... but hides the result of the projection from the consumer
            //if (!this.locks.isEmpty()) {
            //    this.locks.apply((u, p) => {
            //        for (var i = 0; i < this.k; i++) {
            //            r[i][u] = p[i];
            //        }
            //    });
            //}
        }

        private static mApply(m: number, n: number, f: (i: number, j: number) => any) {
            var i = m; while (i-- > 0) {
                var j = n; while (j-- > 0) f(i, j);
            }
        }
        private matrixApply(f: (i: number, j: number) => any) {
            Descent.mApply(this.k, this.n, f);
        }

        private computeNextPosition(x0: Float32Array[], r: Float32Array[]): void {
            this.computeDerivatives(x0);
            const alpha = this.computeStepSize();
            this.stepAndProject(x0, r, this.g, alpha);
/* DEBUG
            for (var u: number = 0; u < this.n; ++u)
                for (var i = 0; i < this.k; ++i)
                    if (isNaN(r[i][u])) debugger;
DEBUG */
            if (this.project) {
                // This functionality is not yet implemented with the Wasm port
                throw new Error('Computing step with with `this.project` set is not yet implemented in Wasm port');
                // this.matrixApply((i, j) => this.e[i][j] = x0[i][j] - r[i][j]);
                // var beta = this.computeStepSize(this.e);
                // beta = Math.max(0.2, Math.min(beta, 1));
                // this.stepAndProject(x0, r, this.e, beta);
            }
        }

        private syncLocks() {
            const entries: number[] = [];
            this.locks.apply((id, p) => {
                entries.push(id, p[0], p[1]);
            });
            if (entries.length === 0 && this.lastLockCount === 0) {
                return;
            }
            this.lastLockCount = entries.length;
            (this.wasm as any).fast_set_locks(this.ctxPtr, new Float32Array(entries));
        }

        public run(iterations: number): number {
            if (this.fast) {
                this.syncLocks();
                return (this.wasm as any).fast_run(this.ctxPtr, iterations, this.threshold, this.integratorMode);
            }

            var stress = Number.MAX_VALUE, converged = false;
            while (!converged && iterations-- > 0) {
                var s = this.rungeKutta();
                converged = Math.abs(stress / s - 1) < this.threshold;
                stress = s;
            }
            return stress;
        }

        public rungeKutta(): number {
            if (this.fast) {
                this.syncLocks();
                let disp = 0;
                for (let i = 0; i < this.substeps; i++) {
                    disp = (this.wasm as any).fast_tick(this.ctxPtr, this.integratorMode);
                }
                return disp;
            }

            this.computeNextPosition(this.x, this.a);
            Descent.mid(this.x, this.a, this.ia);
            this.computeNextPosition(this.ia, this.b);
            Descent.mid(this.x, this.b, this.ib);
            this.computeNextPosition(this.ib, this.c);
            this.computeNextPosition(this.c, this.d);
            var disp = 0;
            this.matrixApply((i, j) => {
                var x = (this.a[i][j] + 2.0 * this.b[i][j] + 2.0 * this.c[i][j] + this.d[i][j]) / 6.0,
                    d = this.x[i][j] - x;
                disp += d * d;
                this.x[i][j] = x;
            });
            return disp;
        }

        private static mid(a: Float32Array[], b: Float32Array[], m: Float32Array[]): void {
            Descent.mApply(a.length, a[0].length, (i, j) =>
                m[i][j] = a[i][j] + (b[i][j] - a[i][j]) / 2.0);
        }

        public takeDescentStep(x: Float32Array, d: Float32Array, stepSize: number): void {
            for (var i = 0; i < this.n; ++i) {
                x[i] = x[i] - stepSize * d[i];
            }
        }

        public computeStress(): number {
            var stress = 0;
            for (var u = 0, nMinus1 = this.n - 1; u < nMinus1; ++u) {
                for (var v = u + 1, n = this.n; v < n; ++v) {
                    var l = 0;
                    for (var i = 0; i < this.k; ++i) {
                        var dx = this.x[i][u] - this.x[i][v];
                        l += dx * dx;
                    }
                    l = Math.sqrt(l);
                    var d = this.D[u][v];
                    if (!isFinite(d)) continue;
                    var rl = d - l;
                    var d2 = d * d;
                    stress += rl * rl / d2;
                }
            }
            return stress;
        }
    }

    // Linear congruential pseudo random number generator
    export class PseudoRandom {
        private a: number = 214013;
        private c: number = 2531011;
        private m: number = 2147483648;
        private range: number = 32767;

        constructor(public seed: number = 1) { }

        // random real between 0 and 1
        getNext(): number {
            this.seed = (this.seed * this.a + this.c) % this.m;
            return (this.seed >> 16) / this.range;
        }

        // random real between min and max
        getNextBetween(min: number, max: number) {
            return min + this.getNext() * (max - min);
        }
    }
