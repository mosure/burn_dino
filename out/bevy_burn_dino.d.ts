/* tslint:disable */
/* eslint-disable */

export function frame_input(pixel_data: Uint8Array, width: number, height: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly frame_input: (a: number, b: number, c: number, d: number) => void;
    readonly main: (a: number, b: number) => number;
    readonly __wasm_bindgen_func_elem_105091: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_111690: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_110922: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_110971: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111686: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111686_5: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111686_6: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111686_7: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_110971_8: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111686_9: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111686_10: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111686_11: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111686_12: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_111698: (a: number, b: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export5: (a: number, b: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
