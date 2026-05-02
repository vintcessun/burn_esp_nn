#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_depthwise_conv_s8_ansi as esp_nn_depthwise_conv_s8;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_depthwise_conv_s8_opt as esp_nn_depthwise_conv_s8;
#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_get_depthwise_conv_scratch_size_ansi as esp_nn_get_depthwise_conv_scratch_size;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_get_depthwise_conv_scratch_size_opt as esp_nn_get_depthwise_conv_scratch_size;
#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_set_depthwise_conv_scratch_buf_ansi as esp_nn_set_depthwise_conv_scratch_buf;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_set_depthwise_conv_scratch_buf_opt as esp_nn_set_depthwise_conv_scratch_buf;

use core::ffi::c_void;

use crate::quant::quantize_multiplier;
use ember_infer_core::{DepthwiseConv2dParams, FusedActivation, KernelError, Padding, Status};
use esp_nn_sys::bindings::{act_params_t, data_2d_t, data_dims_t, dw_conv_params_t, quant_data_t};

// Compile-time stack limit for per-channel quantization. Raise this if a model
// needs more output channels and the target stack budget allows it.
const MAX_CHANNELS: usize = 512;

pub fn run(params: DepthwiseConv2dParams<'_>) -> Status {
    if params.depth_multiplier <= 0 {
        return Err(KernelError::InvalidShape);
    }

    let Some(c_out) = params.input_shape[3].checked_mul(params.depth_multiplier as usize) else {
        return Err(KernelError::InvalidShape);
    };

    if c_out == 0
        || c_out > MAX_CHANNELS
        || c_out != params.output_shape[3]
        || params.weights_shape[3] != c_out
    {
        return Err(KernelError::InvalidShape);
    }

    if !shape4_fits_i32(params.input_shape)
        || !shape4_fits_i32(params.weights_shape)
        || !shape4_fits_i32(params.output_shape)
        || tensor_len(params.input_shape).is_none_or(|len| params.input.len() < len)
        || tensor_len(params.weights_shape).is_none_or(|len| params.weights.len() < len)
        || tensor_len(params.output_shape).is_none_or(|len| params.output.len() < len)
        || params.bias.is_some_and(|bias| bias.len() < c_out)
    {
        return Err(KernelError::InvalidShape);
    }

    let effective_scale =
        (params.input_quant.scale * params.weights_quant.scale) / params.output_quant.scale;
    let (multiplier, shift) = quantize_multiplier(effective_scale as f64);

    let mut mult_arr = [0i32; MAX_CHANNELS];
    let mut shift_arr = [0i32; MAX_CHANNELS];
    for i in 0..c_out {
        mult_arr[i] = multiplier;
        shift_arr[i] = shift;
    }

    let input_dims = dims_from_nhwc(params.input_shape);
    let filter_dims = dims_from_nhwc(params.weights_shape);
    let output_dims = dims_from_nhwc(params.output_shape);
    let (pad_w, pad_h) = match params.padding {
        Padding::Valid => (0, 0),
        Padding::Same => (
            compute_padding(
                params.input_shape[2] as i32,
                params.output_shape[2] as i32,
                params.stride_w,
                params.weights_shape[2] as i32,
            ),
            compute_padding(
                params.input_shape[1] as i32,
                params.output_shape[1] as i32,
                params.stride_h,
                params.weights_shape[1] as i32,
            ),
        ),
    };
    let (activation_min, activation_max) =
        activation_range(params.activation, params.output_quant.zero_point);

    let dw_params = dw_conv_params_t {
        in_offset: -params.input_quant.zero_point,
        out_offset: params.output_quant.zero_point,
        ch_mult: params.depth_multiplier,
        stride: data_2d_t {
            width: params.stride_w,
            height: params.stride_h,
        },
        padding: data_2d_t {
            width: pad_w,
            height: pad_h,
        },
        dilation: data_2d_t {
            width: params.dilation_w_factor,
            height: params.dilation_h_factor,
        },
        activation: act_params_t {
            min: activation_min,
            max: activation_max,
        },
    };
    let q_data = quant_data_t {
        shift: shift_arr.as_mut_ptr(),
        mult: mult_arr.as_mut_ptr(),
    };

    unsafe {
        esp_nn_set_depthwise_conv_scratch_buf(params.scratch.as_mut_ptr().cast::<c_void>());
        esp_nn_depthwise_conv_s8(
            &input_dims,
            params.input.as_ptr(),
            &filter_dims,
            params.weights.as_ptr(),
            params.bias.map_or(core::ptr::null(), |bias| bias.as_ptr()),
            &output_dims,
            params.output.as_mut_ptr(),
            &dw_params,
            &q_data,
        );
    }

    Ok(())
}

pub fn scratch_size(
    input_shape: [usize; 4],
    weights_shape: [usize; 4],
    output_shape: [usize; 4],
) -> usize {
    if !shape4_fits_i32(input_shape)
        || !shape4_fits_i32(weights_shape)
        || !shape4_fits_i32(output_shape)
    {
        return 0;
    }

    let input_dims = dims_from_nhwc(input_shape);
    let filter_dims = dims_from_nhwc(weights_shape);
    let output_dims = dims_from_nhwc(output_shape);
    let dw_params = zero_dw_params();
    let size = unsafe {
        esp_nn_get_depthwise_conv_scratch_size(&input_dims, &filter_dims, &output_dims, &dw_params)
    };

    size.max(0) as usize
}

#[inline]
fn dims_from_nhwc(shape: [usize; 4]) -> data_dims_t {
    data_dims_t {
        width: shape[2] as i32,
        height: shape[1] as i32,
        channels: shape[3] as i32,
        extra: shape[0] as i32,
    }
}

#[inline]
fn zero_dw_params() -> dw_conv_params_t {
    dw_conv_params_t {
        in_offset: 0,
        out_offset: 0,
        ch_mult: 0,
        stride: data_2d_t {
            width: 0,
            height: 0,
        },
        padding: data_2d_t {
            width: 0,
            height: 0,
        },
        dilation: data_2d_t {
            width: 0,
            height: 0,
        },
        activation: act_params_t { min: 0, max: 0 },
    }
}

#[inline]
fn shape4_fits_i32(shape: [usize; 4]) -> bool {
    shape.iter().all(|dim| *dim <= i32::MAX as usize)
}

#[inline]
fn tensor_len(shape: [usize; 4]) -> Option<usize> {
    shape
        .into_iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
}

#[inline]
fn compute_padding(input_dim: i32, output_dim: i32, stride: i32, filter: i32) -> i32 {
    ((output_dim - 1) * stride + filter - input_dim).max(0) / 2
}

#[inline]
fn activation_range(act: FusedActivation, out_zero_point: i32) -> (i32, i32) {
    match act {
        FusedActivation::None => (-128, 127),
        FusedActivation::Relu | FusedActivation::Relu6 => (out_zero_point.max(-128), 127),
        _ => (-128, 127),
    }
}
