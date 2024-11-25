// use burn_fusion::{
//     Fusion,
//     FusionBackend,
//     FusionRuntime,
// };
use burn_jit::{
    kernel::into_contiguous,
    tensor::JitTensor,
    FloatElement,
    IntElement,
    JitBackend,
    JitRuntime,
};
use burn::tensor::{
    ops::FloatTensor,
    Shape,
};
use cubecl::{
    CubeCount,
    CubeDim,
};

use super::Backend;
use super::kernel::adaptive_conv_forward_kernel;


impl<R: JitRuntime, F: FloatElement, I: IntElement> Backend for JitBackend<R, F, I> {
    fn adaptive_conv(
        input: FloatTensor<Self>,    // Input tensor: shape [B, C, H_in, W_in]
        filters: FloatTensor<Self>,  // Filters tensor: shape [B, H_out, W_out, I, J]
    ) -> FloatTensor<Self> {
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        input.assert_is_on_same_device(&filters);

        let input = into_contiguous(input);
        let filters = into_contiguous(filters);

        let B = input.shape.dims[0];
        let C = input.shape.dims[1];
        let H_in = input.shape.dims[2];
        let W_in = input.shape.dims[3];

        let B_filters = filters.shape.dims[0];
        let H_out = filters.shape.dims[1];
        let W_out = filters.shape.dims[2];
        let I = filters.shape.dims[3];
        let J = filters.shape.dims[4];

        assert_eq!(B, B_filters, "Batch size of input and filters must match");
        let H_expected = H_out + I - 1;
        let W_expected = W_out + J - 1;
        assert_eq!(H_in, H_expected, "Input height must be H_out + I - 1");
        assert_eq!(W_in, W_expected, "Input width must be W_out + J - 1");

        let shape_out = Shape::new([B, C, H_out, W_out]);

        let buffer = input
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = JitTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape_out,
            buffer,
        );

        let cubes_needed_in_x = f32::ceil(H_out as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(W_out as f32 / cube_dim.y as f32) as u32;
        let cube_count = CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, B as u32);

        adaptive_conv_forward_kernel::launch::<F, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(1),
            filters.as_tensor_arg(1),
            output.as_tensor_arg(1),
        );

        output
    }
}
