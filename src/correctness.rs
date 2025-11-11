use std::{fmt, fs, path::Path};

use burn::{
    module::Module,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError},
    tensor::Tensor,
};
use safetensors::tensor::{SafeTensors, TensorView};

use crate::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};

#[derive(Debug)]
pub struct CorrectnessReference {
    pub network_input: Vec<f32>,
    pub network_input_shape: [usize; 4],
    pub x_norm_patchtokens: Vec<f32>,
    pub x_norm_patchtokens_shape: [usize; 3],
    pub x_norm_clstoken: Vec<f32>,
    pub x_norm_clstoken_shape: [usize; 2],
    pub x_norm_regtokens: Option<Vec<f32>>,
    pub x_norm_regtokens_shape: Option<[usize; 3]>,
}

impl CorrectnessReference {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CorrectnessError> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;

        let (network_input, network_shape) = read_tensor::<4>(&tensors, "network_input")?;
        let (patchtokens, patch_shape) = read_tensor::<3>(&tensors, "x_norm_patchtokens")?;
        let (clstoken, cls_shape) = read_tensor::<2>(&tensors, "x_norm_clstoken")?;
        let regtokens = read_tensor_optional::<3>(&tensors, "x_norm_regtokens")?;
        let (reg_values, reg_shape) = match regtokens {
            Some((values, shape)) => (Some(values), Some(shape)),
            None => (None, None),
        };

        Ok(Self {
            network_input,
            network_input_shape: network_shape,
            x_norm_patchtokens: patchtokens,
            x_norm_patchtokens_shape: patch_shape,
            x_norm_clstoken: clstoken,
            x_norm_clstoken_shape: cls_shape,
            x_norm_regtokens: reg_values,
            x_norm_regtokens_shape: reg_shape,
        })
    }
}

#[derive(Debug)]
pub struct MetricStats {
    pub mean_abs: f32,
    pub max_abs: f32,
    pub max_rel: f32,
    pub mse: f32,
}

#[derive(Debug)]
pub struct CorrectnessStats {
    pub cls_token: MetricStats,
    pub patch_tokens: MetricStats,
    pub register_tokens: Option<MetricStats>,
}

impl CorrectnessStats {
    pub fn within_defaults(&self) -> bool {
        let patch_ok = self.patch_tokens.max_abs <= PATCH_TOKENS_MAX_ABS
            && self.patch_tokens.mse <= PATCH_TOKENS_MSE
            && self.patch_tokens.mean_abs <= PATCH_TOKENS_MEAN_ABS
            && self.cls_token.max_abs <= CLS_TOKEN_MAX_ABS;

        let register_ok = self
            .register_tokens
            .as_ref()
            .map(|stats| {
                stats.max_abs <= PATCH_TOKENS_MAX_ABS
                    && stats.mse <= PATCH_TOKENS_MSE
                    && stats.mean_abs <= PATCH_TOKENS_MEAN_ABS
            })
            .unwrap_or(true);

        patch_ok && register_ok
    }
}

pub const PATCH_TOKENS_MAX_ABS: f32 = 2e-2;
pub const PATCH_TOKENS_MEAN_ABS: f32 = 3e-3;
pub const PATCH_TOKENS_MSE: f32 = 2e-2;
pub const CLS_TOKEN_MAX_ABS: f32 = 1e-2;

pub fn load_model_from_checkpoint<B: Backend>(
    config: &DinoVisionTransformerConfig,
    checkpoint_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<DinoVisionTransformer<B>, CorrectnessError> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    DinoVisionTransformer::new(device, config.clone())
        .load_file(checkpoint_path.as_ref(), &recorder, device)
        .map_err(CorrectnessError::Recorder)
}

pub fn run_correctness<B: Backend>(
    model: &DinoVisionTransformer<B>,
    reference: &CorrectnessReference,
    device: &B::Device,
) -> Result<CorrectnessStats, CorrectnessError> {
    let outputs = collect_outputs(model, reference, device)?;
    let register_stats = outputs.burn_regtokens.as_ref().and_then(|burn| {
        outputs
            .torch_regtokens
            .as_ref()
            .map(|torch| compute_stats(burn, torch))
    });

    Ok(CorrectnessStats {
        cls_token: compute_stats(&outputs.burn_clstoken, &outputs.torch_clstoken),
        patch_tokens: compute_stats(&outputs.burn_patchtokens, &outputs.torch_patchtokens),
        register_tokens: register_stats,
    })
}

pub struct CorrectnessOutputs {
    pub burn_patchtokens: Vec<f32>,
    pub torch_patchtokens: Vec<f32>,
    pub burn_clstoken: Vec<f32>,
    pub torch_clstoken: Vec<f32>,
    pub burn_regtokens: Option<Vec<f32>>,
    pub torch_regtokens: Option<Vec<f32>>,
}

pub fn collect_outputs<B: Backend>(
    model: &DinoVisionTransformer<B>,
    reference: &CorrectnessReference,
    device: &B::Device,
) -> Result<CorrectnessOutputs, CorrectnessError> {
    let input = Tensor::<B, 1>::from_floats(reference.network_input.as_slice(), device).reshape([
        reference.network_input_shape[0] as i32,
        reference.network_input_shape[1] as i32,
        reference.network_input_shape[2] as i32,
        reference.network_input_shape[3] as i32,
    ]);

    let output = model.forward(input, None);

    let patchtokens = tensor_to_vec(output.x_norm_patchtokens.clone())?;
    let clstoken = tensor_to_vec(output.x_norm_clstoken.clone())?;
    let regtokens = if let Some(reg) = output.x_norm_regtokens.clone() {
        Some(tensor_to_vec(reg)?)
    } else {
        None
    };

    if patchtokens.len() != reference.x_norm_patchtokens.len() {
        return Err(CorrectnessError::LengthMismatch {
            tensor: "x_norm_patchtokens",
            expected: reference.x_norm_patchtokens.len(),
            actual: patchtokens.len(),
        });
    }

    if clstoken.len() != reference.x_norm_clstoken.len() {
        return Err(CorrectnessError::LengthMismatch {
            tensor: "x_norm_clstoken",
            expected: reference.x_norm_clstoken.len(),
            actual: clstoken.len(),
        });
    }

    if let Some(torch_regs) = &reference.x_norm_regtokens {
        match &regtokens {
            Some(burn_regs) => {
                if burn_regs.len() != torch_regs.len() {
                    return Err(CorrectnessError::LengthMismatch {
                        tensor: "x_norm_regtokens",
                        expected: torch_regs.len(),
                        actual: burn_regs.len(),
                    });
                }
            }
            None => {
                return Err(CorrectnessError::MissingTensor("x_norm_regtokens"));
            }
        }
    } else if regtokens.is_some() {
        return Err(CorrectnessError::LengthMismatch {
            tensor: "x_norm_regtokens",
            expected: 0,
            actual: regtokens.as_ref().map(|v| v.len()).unwrap_or_default(),
        });
    }

    Ok(CorrectnessOutputs {
        burn_patchtokens: patchtokens,
        torch_patchtokens: reference.x_norm_patchtokens.clone(),
        burn_clstoken: clstoken,
        torch_clstoken: reference.x_norm_clstoken.clone(),
        burn_regtokens: regtokens,
        torch_regtokens: reference.x_norm_regtokens.clone(),
    })
}

fn compute_stats(burn: &[f32], torch: &[f32]) -> MetricStats {
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mse = 0.0f32;

    for (&lhs, &rhs) in burn.iter().zip(torch.iter()) {
        let diff = lhs - rhs;
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        if rhs.abs() > f32::EPSILON {
            max_rel = max_rel.max(abs / rhs.abs());
        }
        mse += diff.powi(2);
    }

    let len = burn.len() as f32;
    MetricStats {
        mean_abs: sum_abs / len,
        max_abs,
        max_rel,
        mse: mse / len,
    }
}

fn read_tensor<const D: usize>(
    tensors: &SafeTensors<'_>,
    name: &'static str,
) -> Result<(Vec<f32>, [usize; D]), CorrectnessError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| CorrectnessError::MissingTensor(name))?;
    let shape: [usize; D] =
        view.shape()
            .try_into()
            .map_err(|_| CorrectnessError::UnexpectedRank {
                tensor: name,
                expected: D,
                actual: view.shape().len(),
            })?;

    Ok((tensor_view_to_vec(&view), shape))
}

fn read_tensor_optional<const D: usize>(
    tensors: &SafeTensors<'_>,
    name: &'static str,
) -> Result<Option<(Vec<f32>, [usize; D])>, CorrectnessError> {
    match tensors.tensor(name) {
        Ok(view) => {
            let shape: [usize; D] =
                view.shape()
                    .try_into()
                    .map_err(|_| CorrectnessError::UnexpectedRank {
                        tensor: name,
                        expected: D,
                        actual: view.shape().len(),
                    })?;
            Ok(Some((tensor_view_to_vec(&view), shape)))
        }
        Err(_) => Ok(None),
    }
}

fn tensor_view_to_vec(view: &TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(bytes)
        })
        .collect()
}

fn tensor_to_vec<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>, CorrectnessError> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| CorrectnessError::TensorData(format!("{err:?}")))
}

#[derive(Debug)]
pub enum CorrectnessError {
    Io(std::io::Error),
    Safetensors(safetensors::SafeTensorError),
    MissingTensor(&'static str),
    UnexpectedRank {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    TensorData(String),
    Recorder(RecorderError),
}

impl fmt::Display for CorrectnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Safetensors(err) => write!(f, "safetensors error: {err}"),
            Self::MissingTensor(name) => write!(f, "tensor `{name}` missing from reference"),
            Self::UnexpectedRank {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "tensor `{tensor}` rank mismatch: expected {expected}, got {actual}"
            ),
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "tensor `{tensor}` length mismatch: expected {expected}, got {actual}"
            ),
            Self::TensorData(err) => write!(f, "tensor serialization error: {err}"),
            Self::Recorder(err) => write!(f, "recorder error: {err}"),
        }
    }
}

impl std::error::Error for CorrectnessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Safetensors(err) => Some(err),
            Self::Recorder(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CorrectnessError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<safetensors::SafeTensorError> for CorrectnessError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(err)
    }
}
