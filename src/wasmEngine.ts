/**
 * Loads the WebAssembly module that performs the derivative computations for `descent.ts`
 */

import * as wasmSIMD from './wasm/build/simd/derivative_computer';
import wasmSIMD_bg from './wasm/build/simd/derivative_computer_bg.wasm';
import * as wasmNoSIMD from './wasm/build/no_simd/derivative_computer';
import wasmNoSIMD_bg from './wasm/build/no_simd/derivative_computer_bg.wasm';

// prettier-ignore
const getHasSIMDSupport = async () => WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]))

class AsyncOnce<T> {
  private getter: () => Promise<T>;
  private pending: Promise<T> | null = null;
  private res: null | { value: T };

  public constructor(getter: () => Promise<T>) {
    this.getter = getter;
  }

  public async get(): Promise<T> {
    if (this.res) {
      return this.res.value;
    }
    if (this.pending) {
      return this.pending;
    }

    this.pending = new Promise((resolve) =>
      this.getter().then((res) => {
        this.res = { value: res };
        this.pending = null;
        resolve(res);
      })
    );
    return this.pending!;
  }
}

const WasmInst = new AsyncOnce(async () => {
  const hasWasmSIMDSupport = await getHasSIMDSupport();
  if (!window.location.href.includes('localhost')) {
    console.log(
      hasWasmSIMDSupport
        ? 'Wasm SIMD support detected!'
        : 'Wasm SIMD support NOT detected; using non-SIMD Wasm'
    );
  }

  if (hasWasmSIMDSupport) {
    const wasmModule = await (wasmSIMD_bg as any)();
    await (wasmSIMD as any).setWasm(wasmModule);
    return wasmSIMD;
  } else {
    const wasmModule = await (wasmNoSIMD_bg as any)();
    await (wasmNoSIMD as any).setWasm(wasmModule);
    return wasmNoSIMD;
  }
});

type PromiseResolveType<P> = P extends Promise<infer T> ? T : never;

export type DerivativeComputerWasmInst = PromiseResolveType<ReturnType<typeof WasmInst.get>>;

export const getDerivativeComputerWasm = (): Promise<DerivativeComputerWasmInst> => WasmInst.get();
