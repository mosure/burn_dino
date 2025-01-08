use burn::{
    prelude::*,
    module::Param,
    nn::Initializer,
};


#[derive(Config)]
pub struct PcaTransformConfig {
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Default for PcaTransformConfig {
    fn default() -> Self {
        Self::new(384, 3)
    }
}

impl PcaTransformConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PcaTransform<B> {
        PcaTransform::new(device, self)
    }
}



// mod linalg {
//     use burn::{
//         prelude::*,
//         backend::ndarray::{NdArray, NdArrayTensor},
//         tensor::TensorPrimitive,
//     };
//     use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};

//     pub fn tensor_to_array<B: Backend, const D: usize>(
//         tensor: Tensor<B, D>,
//     ) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
//         let arr = Tensor::<NdArray, D>::from_data(tensor.into_data(), &Default::default());
//         let primitive: NdArrayTensor<f32> = arr.into_primitive().tensor();
//         primitive.array.to_owned()
//     }

//     pub fn array_to_tensor<B: Backend, const D: usize>(
//         array: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
//         device: &B::Device,
//     ) -> Tensor<B, D> {
//         let primitive: NdArrayTensor<f32> = NdArrayTensor::new(array.into());
//         let arr = Tensor::<NdArray, D>::from_primitive(TensorPrimitive::Float(primitive));
//         Tensor::<B, D>::from_data(arr.into_data(), device)
//     }

//     pub fn svd(
//         x: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
//     ) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
//         let (u, s, vt) = x.svd(true, true).unwrap();
//         u.dot(&s.diag()).dot(&vt)
//     }
// }


#[derive(Module, Debug)]
pub struct PcaTransform<B: Backend> {
    pub components: Param<Tensor<B, 2>>,
    pub mean: Param<Tensor<B, 2>>,
}

impl<B: Backend> PcaTransform<B> {
    pub fn new(
        device: &B::Device,
        config: &PcaTransformConfig,
    ) -> Self {
        let components = Initializer::Ones.init([config.output_dim, config.input_dim], device);
        let mean = Initializer::Zeros.init([1, config.input_dim], device);

        Self {
            components,
            mean,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let transformed = x.matmul(self.components.val().transpose());
        transformed - self.mean.val().matmul(self.components.val().transpose())
    }

    // pub fn rolling_fit(
    //     &mut self,
    //     x: Tensor<B, 2>,
    //     threshold: f32,
    // ) {

    // }
}
