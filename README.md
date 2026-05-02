# ember-esp-nn

ESP-NN-backed inference components for Ember on Espressif targets.

This repository focuses on the backend inference layer that uses [Espressif ESP-NN](https://github.com/espressif/esp-nn). The `esp_nn_sys` crate is the raw `no_std` binding layer: it generates Rust FFI bindings for ESP-NN, compiles the matching ESP-NN C/assembly sources for selected ESP targets, and exposes the raw ESP-NN symbols to higher-level inference code. The `ember_esp_nn` crate is the higher-level Ember backend crate that implements `ember_infer_core::KernelBackend` on top of those bindings.

## Workspace Layout

```text
crates/
  ember_esp_nn/  Ember KernelBackend implementation backed by ESP-NN
  esp_nn_sys/     Raw FFI bindings and native ESP-NN build script
```

`esp_nn_sys` vendors ESP-NN as a git submodule at:

```text
crates/esp_nn_sys/vendor/esp-nn
```

## Status

- `ember_esp_nn` is a `#![no_std]` backend crate for `ember-infer-core`.
- `ember_esp_nn` currently provides the backend shape, feature forwarding, quantization helper, and operation stubs. Operation calls return `KernelError::InternalError` until the ESP-NN invoke logic is filled in.
- `esp_nn_sys` builds ESP-NN into a static native library with `cc`.
- Rust bindings are generated from ESP-NN headers with `bindgen`.
- Both crates are `#![no_std]`.
- ANSI, ESP32-S3, and ESP32-P4 source selections are supported.
- The final native link is carried through Cargo metadata from the sys crate to downstream binaries.
- `esp_nn_sys` is responsible for ESP-NN bindings; `ember_esp_nn` builds on top of it.

## Supported Backends

| Feature | Intended target | Native compiler | ESP-NN sources |
| --- | --- | --- | --- |
| `ansi` | Generic supported target | target-appropriate GCC | Portable ANSI ESP-NN sources |
| `esp32s3` | `xtensa-esp32s3-none-elf` | `xtensa-esp32s3-elf-gcc` | ANSI + ESP32-S3 optimized sources |
| `esp32p4` | `riscv32imafc-unknown-none-elf` | `riscv32-esp-elf-gcc` | ANSI + ESP32-P4 optimized sources |

If no feature is selected, `esp_nn_sys` builds the ANSI source set.

`ember_esp_nn` forwards the hardware feature flags to `esp_nn_sys`:

| Feature | Effect |
| --- | --- |
| `ansi` | Enables portable ANSI ESP-NN kernels through `esp_nn_sys/ansi` |
| `esp32s3` | Enables ESP32-S3 optimized kernels through `esp_nn_sys/esp32s3` |
| `esp32p4` | Enables ESP32-P4 optimized kernels through `esp_nn_sys/esp32p4` |
| `assume-aligned` | Trust caller-provided buffers are already 16-byte aligned |

## Requirements

- Rust with the target you want to build for.
- The ESP Rust toolchain for Xtensa targets.
- ESP GCC toolchains available on `PATH`:
  - `xtensa-esp-elf-gcc` for ANSI builds targeting Xtensa.
  - `xtensa-esp32s3-elf-gcc` for optimized ESP32-S3 builds.
  - `riscv32-esp-elf-gcc` for ESP32-P4 builds.
- `libclang` available for `bindgen`.

## Clone

Clone with submodules:

```powershell
git clone --recurse-submodules https://github.com/vintcessun/ember-esp-nn.git
```

For an existing clone:

```powershell
git submodule update --init --recursive
```

## Build Checks

Do not suppress warnings with `#[allow(...)]` or equivalent crate-level lints. If a warning is expected during an incremental implementation phase, leave it visible and document why it is acceptable for that phase.

ANSI on RISC-V:

```powershell
cargo clippy -p esp_nn_sys --target riscv32imac-unknown-none-elf --features ansi
```

ANSI on Xtensa:

```powershell
cargo clippy -p esp_nn_sys --target xtensa-esp32s3-none-elf --features ansi
```

ESP32-S3 optimized sources:

```powershell
cargo clippy -p esp_nn_sys --target xtensa-esp32s3-none-elf --features esp32s3
```

ESP32-P4 optimized sources:

```powershell
cargo clippy -p esp_nn_sys --target riscv32imafc-unknown-none-elf --features esp32p4
```

Ember backend checks:

```powershell
cargo check -p ember_esp_nn --target xtensa-esp32s3-none-elf --features esp32s3
cargo check -p ember_esp_nn --target riscv32imafc-unknown-none-elf --features esp32p4
cargo check -p ember_esp_nn --features ansi
```

ESP32-S3 hardware inference test:

```powershell
cargo build -p ember-infer-test
cd tests\ember-infer-test
cargo run
```

The test binary uses `EspBackend` with the sine model:

```text
models/sine.tflite
```

Output is emitted through defmt/RTT. The firmware prints sine `PASS` lines and reports the elapsed runtime when the inference test finishes.

## Build Logging

The build script is quiet by default. To confirm the selected compiler, flags, defines, and ESP-NN source files, set:

```powershell
$env:ESP_NN_SYS_VERBOSE='1'
cargo clippy -p esp_nn_sys --target riscv32imafc-unknown-none-elf --features esp32p4
```

## Regenerating `build.rs`

`crates/esp_nn_sys/build.rs` is generated from `crates/esp_nn_sys/build-template.rs` and the vendored ESP-NN `CMakeLists.txt`.

After changing the template or source extraction logic, run:

```powershell
python scripts\generate_esp_nn_sys_build_rs.py
```

Check that the generated file is up to date:

```powershell
python scripts\generate_esp_nn_sys_build_rs.py --check
```

## Packaging `esp_nn_sys`

From the sys crate directory:

```powershell
cd crates\esp_nn_sys
cargo package --allow-dirty
cargo publish --dry-run --allow-dirty
```

The packaged crate includes the required ESP-NN headers and source files under `vendor/esp-nn`, so crates.io users do not need the workspace-level git submodule.
