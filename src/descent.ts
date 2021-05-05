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
        private wasm: any; // typeof import('./wasm/wasm');
        private ctxPtr: number;

        public threshold: number = 0.0001;
        /**
         * Hessian Matrix
         */
        public H(i: number): Float32Array[] {
            const memory: WebAssembly.Memory = this.wasm.get_memory();
            const memoryView = new Float32Array(memory.buffer);

            let hPtrs: Uint32Array;
            if (this.k === 2) {
                hPtrs = this.wasm.get_H_2d(this.ctxPtr, i);
            } else if (this.k === 3) {
                hPtrs = this.wasm.get_H_3d(this.ctxPtr, i);
            } else {
                throw new Error('Invalid dimensionality');
            }

            return Array.from(hPtrs).map(hPtr => {
                const offset = hPtr / BYTES_PER_F32;
                return memoryView.subarray(offset,offset + this.n);
            });
        }
        /** gradient vector
         * @property g {Float32Array[]}
         */
        public get g(): Float32Array[] {
            const memory: WebAssembly.Memory = this.wasm.get_memory();
            const memoryView = new Float32Array(memory.buffer);

            if (this.k === 2) {
                const offset0 = this.wasm.get_g_2d_0(this.ctxPtr) / BYTES_PER_F32;
                const offset1 = this.wasm.get_g_2d_1(this.ctxPtr) / BYTES_PER_F32;
                return [memoryView.subarray(offset0, offset0 + this.n), memoryView.subarray(offset1, offset1 + this.n)];
            } else if (this.k === 3) {
                const offset0 = this.wasm.get_g_3d_0(this.ctxPtr) / BYTES_PER_F32;
                const offset1 = this.wasm.get_g_3d_1(this.ctxPtr) / BYTES_PER_F32;
                const offset2 = this.wasm.get_g_3d_2(this.ctxPtr) / BYTES_PER_F32;
                return [
                    memoryView.subarray(offset0, offset0 + this.n),
                    memoryView.subarray(offset1, offset1 + this.n),
                    memoryView.subarray(offset2, offset2 + this.n)
                ];
            } else {
                throw new Error('Invalid dimensionality');
            }
        }
        public set G(newG: Float32Array[])  {
            const allG = new Float32Array(this.n * this.n);
            newG.forEach((Gn, i) => allG.set(Gn, i * this.n));

            if (this.k === 2) {
                this.wasm.set_g_2d(this.ctxPtr, allG);
            } else if (this.k === 3) {
                this.wasm.set_g_3d(this.ctxPtr, allG);
            } else {
                throw new Error('Invalid dimensionality');
            }
        }
        /** positions vector
         * @property x {Float32Array[]}
         */
         public get x(): Float32Array[] {
            const memory: WebAssembly.Memory = this.wasm.get_memory();
            const memoryView = new Float32Array(memory.buffer);

            if (this.k === 2) {
                const offset0 = this.wasm.get_x_2d_0(this.ctxPtr) / BYTES_PER_F32;
                const offset1 = this.wasm.get_x_2d_1(this.ctxPtr) / BYTES_PER_F32;
                return [memoryView.subarray(offset0, offset0 + this.n), memoryView.subarray(offset1, offset1 + this.n)];
            } else if (this.k === 3) {
                const offset0 = this.wasm.get_x_3d_0(this.ctxPtr) / BYTES_PER_F32;
                const offset1 = this.wasm.get_x_3d_1(this.ctxPtr) / BYTES_PER_F32;
                const offset2 = this.wasm.get_x_3d_2(this.ctxPtr) / BYTES_PER_F32;
                return [
                    memoryView.subarray(offset0, offset0 + this.n),
                    memoryView.subarray(offset1, offset1 + this.n),
                    memoryView.subarray(offset2, offset2 + this.n)
                ];
            } else {
                throw new Error('Invalid dimensionality');
            }
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
            const memory: WebAssembly.Memory = this.wasm.get_memory();
            const memoryView = new Float32Array(memory.buffer);

            let gPtrs: Uint32Array;
            if (this.k === 2) {
                gPtrs = this.wasm.get_D_2d(this.ctxPtr, );
            } else if (this.k === 3) {
                gPtrs = this.wasm.get_D_3d(this.ctxPtr, );
            } else {
                throw new Error('Invalid dimensionality');
            }

            return Array.from(gPtrs).map(gPtr => {
                const offset = gPtr / BYTES_PER_F32;
                return memoryView.subarray(offset,offset + this.n);
            });
        }

        public computeDerivatives(x: Float32Array[] | null) {
            if (this.k === 2) {
                const packedX = x ? (() => {
                    const packed = new Float32Array(x[0].length * this.k);
                    x.forEach((xn, i) => packed.set(xn, i * this.n));
                    return packed;
                })() : new Float32Array(0);
                const outX = this.wasm.compute_2d(this.ctxPtr, packedX);

                if (x) {
                    x.forEach((xn, i) => {
                        const slice = outX.subarray(i * this.n, i * this.n + this.n);
                        xn.set(slice);
                    })
                }
            } else if (this.k === 3) {
                const packedX = x ? (() => {
                    const packed = new Float32Array(x[0].length * this.k);
                    x.forEach((xn, i) => packed.set(xn, i * this.n));
                    return packed;
                })() : new Float32Array(0);
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
        }

        public locks: Locks;

        private static zeroDistance: number = 1e-10;
        private minD: number;

        // pool of arrays of size n used internally, allocated in constructor
        private Hd: Float32Array[];
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

        private setupWasm(wasm: /* typeof import('./wasm/wasm') */ any, x: number[][], D: number[][], G: number[][] | null = null) {
            this.wasm = wasm;
            // Concat all x into a single vector
            const allX = new Float32Array(this.n * this.k);
            const allD = new Float32Array(this.n * this.n);
            const allG = G ? new Float32Array(this.n * this.k) : new Float32Array(0);
            x.forEach((xn, i) => {
                allX.set(xn, i * this.n);
            });
            D.forEach((dn, i) => {
                allD.set(dn, i * this.n);
            });
            if (G) {
                G.forEach((gn, i) => {
                    allG.set(gn, i * this.n);
                });
            }
            this.ctxPtr = this.wasm.create_derivative_computer_ctx(this.k, this.n, allX, allD, allG);
        }

        /**
         * @method constructor
         * @param x {number[][]} initial coordinates for nodes
         * @param D {number[][]} matrix of desired distances between pairs of nodes
         * @param G {number[][]} [default=null] if specified, G is a matrix of weights for goal terms between pairs of nodes.
         * If G[i][j] > 1 and the separation between nodes i and j is greater than their ideal distance, then there is no contribution for this pair to the goal
         * If G[i][j] <= 1 then it is used as a weighting on the contribution of the variance between ideal and actual separation between i and j to the goal function
         */
        constructor(x: number[][], D: number[][], G: number[][] = null, wasm: /* typeof import('./wasm/wasm') */ any) {
            this.k = x.length; // dimensionality
            var n = this.n = x[0].length; // number of nodes

            // Set up Wasm context
            this.setupWasm(wasm, x, D, G);

            this.Hd = new Array(this.k);
            this.a = new Array(this.k);
            this.b = new Array(this.k);
            this.c = new Array(this.k);
            this.d = new Array(this.k);
            this.e = new Array(this.k);
            this.ia = new Array(this.k);
            this.ib = new Array(this.k);
            this.xtmp = new Array(this.k);
            this.locks = new Locks();
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
                this.Hd[i] = new Float32Array(n);
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
        public computeStepSize(d: Float32Array[]): number {
            var numerator = 0, denominator = 0;
            for (var i = 0; i < this.k; ++i) {
                numerator += Descent.dotProd(this.g[i], d[i]);
                Descent.rightMultiply(this.H(i), d[i], this.Hd[i]);
                denominator += Descent.dotProd(d[i], this.Hd[i]);
            }
            if (denominator === 0 || !isFinite(denominator)) return 0;
            return 1 * numerator / denominator;
        }

        public reduceStress(): number {
            this.computeDerivatives(null);
            var alpha = this.computeStepSize(this.g);
            for (var i = 0; i < this.k; ++i) {
                this.takeDescentStep(this.x[i], this.g[i], alpha);
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
            // TODO TODO TODO
            this.computeDerivatives(x0);
            var alpha = this.computeStepSize(this.g);
            this.stepAndProject(x0, r, this.g, alpha);
/* DEBUG
            for (var u: number = 0; u < this.n; ++u)
                for (var i = 0; i < this.k; ++i)
                    if (isNaN(r[i][u])) debugger;
DEBUG */
            if (this.project) {
                this.matrixApply((i, j) => this.e[i][j] = x0[i][j] - r[i][j]);
                var beta = this.computeStepSize(this.e);
                beta = Math.max(0.2, Math.min(beta, 1));
                this.stepAndProject(x0, r, this.e, beta);
            }
        }

        public run(iterations: number): number {
            var stress = Number.MAX_VALUE, converged = false;
            while (!converged && iterations-- > 0) {
                var s = this.rungeKutta();
                converged = Math.abs(stress / s - 1) < this.threshold;
                stress = s;
            }
            return stress;
        }

        public rungeKutta(): number {
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
