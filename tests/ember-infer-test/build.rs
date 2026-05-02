fn main() {
    linker_be_nice();
    compile_printf_slim();
    println!("cargo:rustc-link-arg-tests=-Tembedded-test.x");
    println!("cargo:rustc-link-arg=-Tdefmt.x");
    // make sure linkall.x is the last linker script (otherwise might cause problems with flip-link)
    println!("cargo:rustc-link-arg=-Tlinkall.x");
}

fn compile_printf_slim() {
    let out_dir =
        std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let object = out_dir.join("printf_slim.o");

    println!("cargo:rerun-if-changed=src/printf_slim.c");

    let mut build = cc::Build::new();
    if std::env::var("TARGET")
        .map(|target| target == "xtensa-esp32s3-none-elf")
        .unwrap_or(false)
    {
        build.compiler("xtensa-esp32s3-elf-gcc");
    }

    let status = build
        .warnings(false)
        .get_compiler()
        .to_command()
        .arg("-c")
        .arg("src/printf_slim.c")
        .arg("-o")
        .arg(&object)
        .status()
        .expect("failed to invoke C compiler for printf_slim.c");

    if !status.success() {
        panic!("failed to compile printf_slim.c");
    }

    println!("cargo:rustc-link-arg={}", object.display());
}

fn linker_be_nice() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let kind = &args[1];
        let what = &args[2];

        match kind.as_str() {
            "undefined-symbol" => match what.as_str() {
                what if what.starts_with("_defmt_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `defmt` not found - make sure `defmt.x` is added as a linker script and you have included `use defmt_rtt as _;`"
                    );
                    eprintln!();
                }
                "_stack_start" => {
                    eprintln!();
                    eprintln!("💡 Is the linker script `linkall.x` missing?");
                    eprintln!();
                }
                what if what.starts_with("esp_rtos_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `esp-radio` has no scheduler enabled. Make sure you have initialized `esp-rtos` or provided an external scheduler."
                    );
                    eprintln!();
                }
                "embedded_test_linker_file_not_added_to_rustflags" => {
                    eprintln!();
                    eprintln!(
                        "💡 `embedded-test` not found - make sure `embedded-test.x` is added as a linker script for tests"
                    );
                    eprintln!();
                }
                "free"
                | "malloc"
                | "calloc"
                | "get_free_internal_heap_size"
                | "malloc_internal"
                | "realloc_internal"
                | "calloc_internal"
                | "free_internal" => {
                    eprintln!();
                    eprintln!(
                        "💡 Did you forget the `esp-alloc` dependency or didn't enable the `compat` feature on it?"
                    );
                    eprintln!();
                }
                _ => (),
            },
            // we don't have anything helpful for "missing-lib" yet
            _ => {
                std::process::exit(1);
            }
        }

        std::process::exit(0);
    }

    println!(
        "cargo:rustc-link-arg=-Wl,--error-handling-script={}",
        std::env::current_exe().unwrap().display()
    );
}
