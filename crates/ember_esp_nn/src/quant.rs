/// Convert a scale factor to the (multiplier, shift) pair used by ESP-NN's
/// quant_data_t.
///
/// Mirrors TFLite Micro's QuantizeMultiplier using IEEE 754 bit manipulation:
/// - Extract base-2 exponent from the f64 bit pattern.
/// - Reconstruct mantissa in [2^52, 2^53), shift to the Q0.31 range.
/// - Apply round-to-nearest (add LSB before final right shift).
/// - Return (multiplier, exponent) where shift > 0 means left-shift,
///   shift < 0 means right-shift (ESP-NN / TFLite Micro convention).
///
/// Special cases:
/// - scale == 0.0  → (0, 0)
/// - subnormal     → (0, 0)  (negligibly small, safe approximation)
pub fn quantize_multiplier(scale: f64) -> (i32, i32) {
    if scale == 0.0 {
        return (0, 0);
    }

    let bits = scale.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);

    // subnormal — scale is negligibly small
    if exp == 0 {
        return (0, 0);
    }

    // Unbiased base-2 exponent. The reconstructed mantissa is in [1, 2), while
    // QuantizeMultiplier uses frexp's mantissa in [0.5, 1), so the returned
    // shift is one larger than the raw IEEE exponent.
    let exponent = exp - 1023;

    // reconstruct full mantissa with implicit leading 1: [2^52, 2^53)
    let mantissa = (1_u64 << 52) | fraction;

    // shift to [2^31, 2^32), then round-to-nearest into [2^30, 2^31)
    // (add LSB of scaled before the final >> 1 to implement rounding)
    let scaled = mantissa >> 21; // [2^31, 2^32)
    let rounded = (scaled >> 1) + (scaled & 1); // round, → [2^30, 2^31)

    (clamp_i32(rounded), exponent + 1)
}

fn clamp_i32(value: u64) -> i32 {
    if value > i32::MAX as u64 {
        i32::MAX
    } else {
        value as i32
    }
}

#[cfg(test)]
mod tests {
    use super::quantize_multiplier;

    #[test]
    fn multiplier_shift_reconstruct_sine_layer_scales() {
        let scales = [
            (0.024573976_f64 * 0.004107884_f64) / 0.011006054_f64,
            (0.011006054_f64 * 0.009258529_f64) / 0.008667699_f64,
            (0.008667699_f64 * 0.012458475_f64) / 0.008269669_f64,
        ];

        for scale in scales {
            let (mult, shift) = quantize_multiplier(scale);
            let reconstructed = (mult as f64) * 2_f64.powi(shift) / ((1_i64 << 31) as f64);
            let relative_error = ((reconstructed - scale) / scale).abs();
            assert!(
                relative_error < 1.0e-8,
                "scale={scale} mult={mult} shift={shift} reconstructed={reconstructed}"
            );
        }
    }
}
