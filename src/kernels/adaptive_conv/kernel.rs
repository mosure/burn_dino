use cubecl::{
    prelude::*,
    cube,
};


#[cube(launch)]
pub fn adaptive_conv_forward_kernel<F: Float>(
    input: &Tensor<F>,      // Input tensor: shape [B, C, H_in, W_in]
    filters: &Tensor<F>,    // Filters tensor: shape [B, H_out, W_out, I, J]
    output: &mut Tensor<F>, // Output tensor: shape [B, C, H_out, W_out]
) {
    let h = ABSOLUTE_POS_X;
    let w = ABSOLUTE_POS_Y;
    let batch = ABSOLUTE_POS_Z;

    let B = output.shape(0);
    let C = output.shape(1);
    let H_out = output.shape(2);
    let W_out = output.shape(3);

    if batch >= B || h >= H_out || w >= W_out {
        return;
    }

    let H_in = input.shape(2);
    let W_in = input.shape(3);

    let I = filters.shape(3);
    let J = filters.shape(4);

    let stride_input_b = input.stride(0);
    let stride_input_c = input.stride(1);
    let stride_input_h = input.stride(2);
    let stride_input_w = input.stride(3);

    let stride_filters_b = filters.stride(0);
    let stride_filters_h = filters.stride(1);
    let stride_filters_w = filters.stride(2);
    let stride_filters_i = filters.stride(3);
    let stride_filters_j = filters.stride(4);

    let stride_output_b = output.stride(0);
    let stride_output_c = output.stride(1);
    let stride_output_h = output.stride(2);
    let stride_output_w = output.stride(3);

    for c in 0..C {
        let mut output_val = F::new(0.0);
        for i in 0..I {
            for j in 0..J {
                let input_h = h + i;
                let input_w = w + j;

                if input_h < H_in && input_w < W_in {
                    let idx_filter = batch * stride_filters_b
                                    + h * stride_filters_h
                                    + w * stride_filters_w
                                    + i * stride_filters_i
                                    + j * stride_filters_j;

                    let idx_input = batch * stride_input_b
                                    + c * stride_input_c
                                    + input_h * stride_input_h
                                    + input_w * stride_input_w;

                    let weight = filters[idx_filter];
                    let input_val = input[idx_input];

                    output_val += weight * input_val;
                }
            }
        }

        let idx_output = batch * stride_output_b
                        + c * stride_output_c
                        + h * stride_output_h
                        + w * stride_output_w;

        output[idx_output] = output_val;
    }
}
