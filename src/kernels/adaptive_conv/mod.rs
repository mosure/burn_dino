use burn::tensor::{ops::FloatTensor, Tensor, TensorPrimitive};

pub mod forward;
pub mod kernel;


pub trait Backend: burn::tensor::backend::Backend {
    fn adaptive_conv(
        input: FloatTensor<Self>,    // Input tensor: shape [B, C, H_in, W_in]
        filters: FloatTensor<Self>,  // Filters tensor: shape [B, H_out, W_out, I, J]
    ) -> FloatTensor<Self>;
}

pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}


pub fn adaptive_conv<B: Backend>(
    input: Tensor<B, 4>,    // Input tensor: shape [B, C, H_in, W_in]
    filters: Tensor<B, 5>,  // Filters tensor: shape [B, H_out, W_out, I, J]
) -> Tensor<B, 4> {
    let output = B::adaptive_conv(
        input.into_primitive().tensor(),
        filters.into_primitive().tensor(),
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}
