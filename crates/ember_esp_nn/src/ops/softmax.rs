#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_get_softmax_scratch_size_ansi as esp_nn_get_softmax_scratch_size;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_get_softmax_scratch_size_opt as esp_nn_get_softmax_scratch_size;
#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_set_softmax_scratch_buf_ansi as esp_nn_set_softmax_scratch_buf;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_set_softmax_scratch_buf_opt as esp_nn_set_softmax_scratch_buf;
#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_softmax_s8_ansi as esp_nn_softmax_s8;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_softmax_s8_opt as esp_nn_softmax_s8;

use core::ffi::c_void;

use crate::quant::quantize_multiplier;
use ember_infer_core::{KernelError, SoftmaxParams, Status};

const INPUT_INTEGER_BITS: i32 = 5;

pub fn run(params: SoftmaxParams<'_>) -> Status {
    let batch = params.input_shape[0];
    let num_classes = params.input_shape[1];

    if batch > i32::MAX as usize
        || num_classes > i32::MAX as usize
        || batch
            .checked_mul(num_classes)
            .is_none_or(|len| params.input.len() < len || params.output.len() < len)
    {
        return Err(KernelError::InvalidShape);
    }

    let scale = params.beta as f64
        * params.input_quant.scale as f64
        * ((1_i64 << (31 - INPUT_INTEGER_BITS)) as f64);
    let (mult, shift) = quantize_multiplier(scale);
    let diff_min = -calculate_input_radius(INPUT_INTEGER_BITS, shift);

    unsafe {
        esp_nn_set_softmax_scratch_buf(params.scratch.as_mut_ptr().cast::<c_void>());
        esp_nn_softmax_s8(
            params.input.as_ptr(),
            batch as i32,
            num_classes as i32,
            mult,
            shift,
            diff_min,
            params.output.as_mut_ptr(),
        );
    }

    Ok(())
}

pub fn scratch_size(num_classes: usize) -> usize {
    if num_classes > i32::MAX as usize {
        return 0;
    }

    let size = unsafe { esp_nn_get_softmax_scratch_size(num_classes as i32, 1) };
    size.max(0) as usize
}

#[inline]
fn calculate_input_radius(input_integer_bits: i32, input_left_shift: i32) -> i32 {
    let total_signed_bits = 31;
    let max_input_rescaled =
        ((1_i64 << input_integer_bits) - 1) * (1_i64 << (total_signed_bits - input_integer_bits));

    if input_left_shift <= 0 {
        max_input_rescaled as i32
    } else {
        (max_input_rescaled >> input_left_shift) as i32
    }
}
