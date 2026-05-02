mod add;
mod conv2d;
mod depthwise;
mod fully_connected;
mod pool;
mod softmax;

pub struct EspBackend;

use ember_infer_core::{
    Conv2dParams, DepthwiseConv2dParams, ElementwiseAddParams, FullyConnectedParams, KernelBackend,
    PoolParams, SoftmaxParams, Status,
};

impl KernelBackend for EspBackend {
    fn conv2d(&mut self, params: Conv2dParams<'_>) -> Status {
        conv2d::run(params)
    }

    fn depthwise_conv2d(&mut self, params: DepthwiseConv2dParams<'_>) -> Status {
        depthwise::run(params)
    }

    fn fully_connected(&mut self, params: FullyConnectedParams<'_>) -> Status {
        fully_connected::run(params)
    }

    fn avg_pool(&mut self, params: PoolParams<'_>) -> Status {
        pool::run_avg(params)
    }

    fn max_pool(&mut self, params: PoolParams<'_>) -> Status {
        pool::run_max(params)
    }

    fn softmax(&mut self, params: SoftmaxParams<'_>) -> Status {
        softmax::run(params)
    }

    fn add(&mut self, params: ElementwiseAddParams<'_>) -> Status {
        add::run(params)
    }

    fn conv2d_scratch_size(
        input_shape: [usize; 4],
        weights_shape: [usize; 4],
        output_shape: [usize; 4],
    ) -> usize {
        conv2d::scratch_size(input_shape, weights_shape, output_shape)
    }

    fn depthwise_conv2d_scratch_size(
        input_shape: [usize; 4],
        weights_shape: [usize; 4],
        output_shape: [usize; 4],
    ) -> usize {
        depthwise::scratch_size(input_shape, weights_shape, output_shape)
    }

    fn softmax_scratch_size(num_classes: usize) -> usize {
        softmax::scratch_size(num_classes)
    }
}
