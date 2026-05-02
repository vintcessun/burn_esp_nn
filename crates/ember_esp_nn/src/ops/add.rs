#[cfg(feature = "esp32s3")]
use esp_nn_sys::bindings::esp_nn_add_elementwise_s8_esp32s3 as esp_nn_add_elementwise_s8;
#[cfg(not(feature = "esp32s3"))]
use esp_nn_sys::bindings::esp_nn_add_elementwise_s8_ansi as esp_nn_add_elementwise_s8;

use crate::quant::quantize_multiplier;
use ember_infer_core::{ElementwiseAddParams, FusedActivation, KernelError, Status};

const LEFT_SHIFT: i32 = 20;

pub fn run(params: ElementwiseAddParams<'_>) -> Status {
    if params.input1.len() > i32::MAX as usize
        || params.input2.len() < params.input1.len()
        || params.output.len() < params.input1.len()
    {
        return Err(KernelError::InvalidShape);
    }

    let input1_scale = params.input1_quant.scale as f64;
    let input2_scale = params.input2_quant.scale as f64;
    let output_scale = params.output_quant.scale as f64;

    let twice_max_input_scale = 2.0 * input1_scale.max(input2_scale);
    let real_input1_multiplier = input1_scale / twice_max_input_scale;
    let real_input2_multiplier = input2_scale / twice_max_input_scale;
    let real_output_multiplier =
        twice_max_input_scale / ((1_i64 << LEFT_SHIFT) as f64 * output_scale);

    let (input1_mult, input1_shift) = quantize_multiplier(real_input1_multiplier);
    let (input2_mult, input2_shift) = quantize_multiplier(real_input2_multiplier);
    let (out_mult, out_shift) = quantize_multiplier(real_output_multiplier);
    let (activation_min, activation_max) =
        activation_range(params.activation, params.output_quant.zero_point);

    unsafe {
        esp_nn_add_elementwise_s8(
            params.input1.as_ptr(),
            params.input2.as_ptr(),
            -params.input1_quant.zero_point,
            -params.input2_quant.zero_point,
            input1_mult,
            input2_mult,
            input1_shift,
            input2_shift,
            LEFT_SHIFT,
            params.output.as_mut_ptr(),
            params.output_quant.zero_point,
            out_mult,
            out_shift,
            activation_min,
            activation_max,
            params.input1.len() as i32,
        );
    }

    Ok(())
}

#[inline]
fn activation_range(act: FusedActivation, out_zero_point: i32) -> (i32, i32) {
    match act {
        FusedActivation::None => (-128, 127),
        FusedActivation::Relu | FusedActivation::Relu6 => (out_zero_point.max(-128), 127),
        _ => (-128, 127),
    }
}
