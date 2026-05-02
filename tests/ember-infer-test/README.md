# ember-infer-test

ESP32-S3 hardware inference smoke tests for `ember_esp_nn`.

The firmware keeps the generated ESP `defmt`/RTT/embassy project structure. Test output is emitted with `defmt::info!`; ESP-NN's internal `printf` references are satisfied locally by `src/printf_slim.c`.

Run the sine model test:

```powershell
cd tests\ember-infer-test
cargo run
```

The sine test prints `PASS` lines and a final elapsed runtime line over defmt/RTT.
