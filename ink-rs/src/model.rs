use crate::flatbuffer::{Reader, Table, Vector};
use crate::kernels::{self, Backend, PreparedInput, Projection};
use crate::session::{Session, Worker};
use crate::{error, Result};

pub struct Model {
    layers: Vec<Layer>,
    projection: Projection,
    bias: Vec<f32>,
    input_width: usize,
}

struct Layer {
    forward: Direction,
    backward: Direction,
}

struct Direction {
    projection: Projection,
    recurrent: [Vec<f32>; 4],
    bias: [Vec<f32>; 4],
    units: usize,
}

#[derive(Default)]
pub(crate) struct DirectionScratch {
    pre: Vec<f32>,
    hidden: Vec<f32>,
    cell: Vec<f32>,
    pub output: Vec<f32>,
}

#[derive(Default)]
pub(crate) struct Workspace {
    prepared: std::sync::Arc<PreparedInput>,
    forward: DirectionScratch,
    backward: DirectionScratch,
    pub logits: Vec<f32>,
}

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn tensor(index: i32, tensors: Vector<'_>, buffers: Vector<'_>) -> Result<Tensor> {
    let index = usize::try_from(index).map_err(|_| error("Missing model tensor"))?;
    let table = tensors.table(index)?;
    let shape = table.required_vector(0, 4)?;
    let shape = (0..shape.len)
        .map(|i| usize::try_from(shape.i32(i)?).map_err(|_| error("Negative weight dimension")))
        .collect::<Result<Vec<_>>>()?;
    let size = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| error("Tensor shape overflow"))?;
    let buffer = buffers.table(table.u32(2, 0)? as usize)?;
    let bytes = buffer
        .vector(0, 1)?
        .map(|v| v.bytes())
        .transpose()?
        .unwrap_or(&[]);
    if bytes.is_empty() {
        // The four initial recurrent state tensors are zero-filled variables.
        return Ok(Tensor {
            shape,
            data: vec![0.0; size],
        });
    }
    let data: Vec<f32> = match table.i8(1, 0)? {
        0 => {
            if size.checked_mul(4) != Some(bytes.len()) {
                return Err(error("Float tensor buffer length mismatch"));
            }
            bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        }
        3 => {
            // These custom-op packs label symmetric signed int8 kernels UINT8.
            if bytes.len() != size {
                return Err(error("Quantized tensor buffer length mismatch"));
            }
            let q = table
                .table(4)?
                .ok_or_else(|| error("Missing weight quantization"))?;
            let scales = q.required_vector(2, 4)?;
            if scales.len != 1 {
                return Err(error("Expected per-tensor weight quantization"));
            }
            let scale = scales.f32(0)?;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(error("Invalid weight scale"));
            }
            if let Some(zeros) = q.vector(3, 8)? {
                if zeros.len != 1 || zeros.i64(0)? != 0 {
                    return Err(error("Expected symmetric signed weights"));
                }
            }
            bytes.iter().map(|v| (*v as i8) as f32 * scale).collect()
        }
        kind => return Err(error(format!("Unsupported weight tensor type {kind}"))),
    };
    if data.iter().any(|v: &f32| !v.is_finite()) {
        return Err(error("Non-finite model weight"));
    }
    Ok(Tensor { shape, data })
}

fn direction(
    inputs: &[i32],
    offset: usize,
    width: usize,
    tensors: Vector<'_>,
    buffers: Vector<'_>,
) -> Result<Direction> {
    let mut weights = Vec::new();
    let mut units = 0;
    for gate in 0..4 {
        let w = tensor(inputs[offset + gate], tensors, buffers)?;
        if w.shape.len() != 2 || w.shape[1] != width || w.shape[0] == 0 {
            return Err(error("Invalid IndyLSTM input weights"));
        }
        if gate == 0 {
            units = w.shape[0];
        }
        if w.shape[0] != units {
            return Err(error("Gate unit counts differ"));
        }
        weights.extend(w.data);
    }
    let mut recurrent: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::new());
    let mut bias: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::new());
    for gate in 0..4 {
        let u = tensor(inputs[offset + 4 + gate], tensors, buffers)?;
        let b = tensor(inputs[offset + 8 + gate], tensors, buffers)?;
        if u.shape != [units] || b.shape != [units] {
            return Err(error("Invalid recurrent vector or gate bias"));
        }
        recurrent[gate] = u.data;
        bias[gate] = b.data;
    }
    Ok(Direction {
        projection: Projection::new(width, 4 * units, &weights)?,
        recurrent,
        bias,
        units,
    })
}

impl Model {
    /// Read the existing TFLite pack without an Android or LiteRT runtime.
    /// Supports the validated IndyLSTM operator configuration and final FC.
    pub fn from_tflite(bytes: &[u8]) -> Result<Self> {
        let root = Reader { bytes }.root()?;
        if root.u32(0, 0)? != 3 {
            return Err(error("Expected TFLite schema version 3"));
        }
        let graphs = root.required_vector(2, 4)?;
        if graphs.len != 1 {
            return Err(error("Expected one handwriting subgraph"));
        }
        let graph = graphs.table(0)?;
        let tensors = graph.required_vector(0, 4)?;
        let buffers = root.required_vector(4, 4)?;
        let codes = root.required_vector(1, 4)?;
        let operations = graph.required_vector(3, 4)?;
        let graph_inputs = graph.required_vector(1, 4)?;
        let graph_outputs = graph.required_vector(2, 4)?;
        if graph_inputs.len != 1 || graph_outputs.len != 1 {
            return Err(error("Expected one graph input/output"));
        }
        let mut previous_tensor = graph_inputs.i32(0)?;
        let input_width = 10;
        let mut width = input_width;
        let mut layers = Vec::new();
        let mut final_projection = None;
        for index in 0..operations.len {
            let op = operations.table(index)?;
            let ins = op.required_vector(1, 4)?;
            let outs = op.required_vector(2, 4)?;
            if ins.len == 0 || outs.len != 1 || ins.i32(0)? != previous_tensor {
                return Err(error("Unsupported handwriting graph connectivity"));
            }
            let inputs = (0..ins.len)
                .map(|i| ins.i32(i))
                .collect::<Result<Vec<_>>>()?;
            let code = codes.table(op.u32(0, 0)? as usize)?;
            let kind = builtin(code)?;
            if kind == 32 && code.string(1)? == Some("bidirectional_sequence_indylstm") {
                if final_projection.is_some() || inputs.len() != 29 {
                    return Err(error("Unsupported IndyLSTM inputs"));
                }
                let options = op.required_vector(5, 1)?.bytes()?;
                if options != [0, 0, 72, 66, 4, 1, 1, 0] {
                    return Err(error("Unsupported IndyLSTM custom options"));
                }
                let forward = direction(&inputs, 1, width, tensors, buffers)?;
                let backward = direction(&inputs, 13, width, tensors, buffers)?;
                if forward.units != backward.units {
                    return Err(error("Bidirectional unit counts differ"));
                }
                for &state in &inputs[25..29] {
                    let state = tensor(state, tensors, buffers)?;
                    if state.shape != [1, forward.units] || state.data.iter().any(|v| *v != 0.0) {
                        return Err(error("Expected zero initial recurrent states"));
                    }
                }
                width = 2 * forward.units;
                layers.push(Layer { forward, backward });
            } else if kind == 9 && inputs.len() == 3 && index + 1 == operations.len {
                let w = tensor(inputs[1], tensors, buffers)?;
                let b = tensor(inputs[2], tensors, buffers)?;
                if w.shape.len() != 2 || w.shape[1] != width || b.shape != [w.shape[0]] {
                    return Err(error("Invalid final projection"));
                }
                final_projection = Some((Projection::new(width, w.shape[0], &w.data)?, b.data));
            } else {
                return Err(error(format!("Unsupported model operator {kind}")));
            }
            previous_tensor = outs.i32(0)?;
        }
        if layers.is_empty() || previous_tensor != graph_outputs.i32(0)? {
            return Err(error("Incomplete handwriting graph"));
        }
        let (projection, bias) =
            final_projection.ok_or_else(|| error("Missing final projection"))?;
        Ok(Self {
            layers,
            projection,
            bias,
            input_width,
        })
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_tflite(&std::fs::read(path)?)
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }
    pub fn output_width(&self) -> usize {
        self.projection.outputs
    }
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Fresh recurrent state for every call. For repeated calls, use a Session
    /// to retain scratch capacity and optionally use one backward worker.
    pub fn infer(&self, features: &[f32], timesteps: usize, backend: Backend) -> Result<Vec<f32>> {
        let backend = backend.resolve()?;
        let mut workspace = Workspace::default();
        self.infer_with_workspace(features, timesteps, backend, &mut workspace, None)?;
        Ok(workspace.logits)
    }

    /// At most two compute threads: the caller and one persistent worker.
    pub fn into_session(self, backend: Backend, threads: usize) -> Result<Session> {
        Session::new(self, backend, threads)
    }

    pub(crate) fn infer_with_workspace(
        &self,
        features: &[f32],
        timesteps: usize,
        backend: Backend,
        workspace: &mut Workspace,
        worker: Option<&Worker>,
    ) -> Result<()> {
        if timesteps == 0 || timesteps.checked_mul(self.input_width) != Some(features.len()) {
            return Err(error("Expected nonempty [T,10] curve features"));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(error("Non-finite input feature"));
        }
        let prepared = std::sync::Arc::get_mut(&mut workspace.prepared)
            .ok_or_else(|| error("Packed input is still in use"))?;
        prepared.prepare(features, timesteps, self.input_width, backend);
        for (index, layer) in self.layers.iter().enumerate() {
            if let Some(worker) = worker {
                worker.submit(index, std::sync::Arc::clone(&workspace.prepared))?;
                layer
                    .forward
                    .run_into(&workspace.prepared, false, &mut workspace.forward);
                let backward = worker.finish()?;
                let prepared = std::sync::Arc::get_mut(&mut workspace.prepared)
                    .ok_or_else(|| error("Packed input is still in use"))?;
                prepared.prepare_directions(
                    &workspace.forward.output,
                    &backward.output,
                    timesteps,
                    layer.forward.units,
                    backend,
                );
            } else {
                layer
                    .forward
                    .run_into(&workspace.prepared, false, &mut workspace.forward);
                layer
                    .backward
                    .run_into(&workspace.prepared, true, &mut workspace.backward);
                let prepared = std::sync::Arc::get_mut(&mut workspace.prepared)
                    .ok_or_else(|| error("Packed input is still in use"))?;
                prepared.prepare_directions(
                    &workspace.forward.output,
                    &workspace.backward.output,
                    timesteps,
                    layer.forward.units,
                    backend,
                );
            }
        }
        self.projection
            .forward_into(&workspace.prepared, &mut workspace.logits);
        for row in workspace.logits.chunks_exact_mut(self.output_width()) {
            for (value, bias) in row.iter_mut().zip(&self.bias) {
                *value += bias;
            }
        }
        Ok(())
    }

    pub(crate) fn run_backward(
        &self,
        layer: usize,
        prepared: &PreparedInput,
        scratch: &mut DirectionScratch,
    ) {
        self.layers[layer]
            .backward
            .run_into(prepared, true, scratch);
    }
}

fn builtin(code: Table<'_>) -> Result<i32> {
    let modern = code.i32(3, 0)?;
    Ok(if modern >= 127 {
        modern
    } else {
        i32::from(code.i8(0, 0)?)
    })
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

impl Direction {
    #[cfg(test)]
    fn run(&self, x: &[f32], timesteps: usize, reverse: bool, backend: Backend) -> Vec<f32> {
        let mut prepared = PreparedInput::default();
        prepared.prepare(x, timesteps, self.projection.inputs, backend);
        let mut scratch = DirectionScratch::default();
        self.run_into(&prepared, reverse, &mut scratch);
        scratch.output
    }

    fn run_into(&self, prepared: &PreparedInput, reverse: bool, scratch: &mut DirectionScratch) {
        self.projection.forward_into(prepared, &mut scratch.pre);
        let timesteps = prepared.rows;
        let backend = prepared.backend;
        scratch.hidden.resize(self.units, 0.0);
        scratch.cell.resize(self.units, 0.0);
        scratch.output.resize(timesteps * self.units, 0.0);
        scratch.hidden.fill(0.0);
        scratch.cell.fill(0.0);
        let DirectionScratch {
            pre,
            hidden,
            cell,
            output,
        } = scratch;
        for step in 0..timesteps {
            let t = if reverse { timesteps - 1 - step } else { step };
            let frame = &pre[t * 4 * self.units..(t + 1) * 4 * self.units];
            let vector_end = if backend == Backend::Neon {
                self.units / 4 * 4
            } else {
                0
            };
            for j in (0..vector_end).step_by(4) {
                let mut gates = [[0.0; 4]; 4];
                for (gate, values) in gates.iter_mut().enumerate() {
                    kernels::affine4(
                        &frame[gate * self.units + j..],
                        &self.bias[gate][j..],
                        &self.recurrent[gate][j..],
                        &hidden[j..],
                        values,
                    );
                    kernels::activation4(values, gate == 2);
                }
                let mut c = [0.0; 4];
                kernels::cell4(&gates[1], &cell[j..], &gates[0], &gates[2], &mut c);
                cell[j..j + 4].copy_from_slice(&c);
                kernels::activation4(&mut c, true);
                let mut h = [0.0; 4];
                kernels::mul4(&gates[3], &c, &mut h);
                hidden[j..j + 4].copy_from_slice(&h);
            }
            for j in vector_end..self.units {
                let mut values = [0.0; 4];
                for gate in 0..4 {
                    let a = (frame[gate * self.units + j] + self.bias[gate][j])
                        + self.recurrent[gate][j] * hidden[j];
                    values[gate] = if gate == 2 { a.tanh() } else { sigmoid(a) };
                }
                cell[j] = values[1] * cell[j] + values[0] * values[2];
                hidden[j] = values[3] * cell[j].tanh();
            }
            output[t * self.units..(t + 1) * self.units].copy_from_slice(hidden);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_model() -> Model {
        let units = 5;
        let make_direction = || Direction {
            projection: Projection::new(
                10,
                units * 4,
                &(0..units * 40)
                    .map(|i| ((i * 7 % 31) as f32 - 15.0) / 32.0)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            recurrent: std::array::from_fn(|_| vec![0.25; units]),
            bias: std::array::from_fn(|_| vec![0.125; units]),
            units,
        };
        Model {
            layers: vec![Layer {
                forward: make_direction(),
                backward: make_direction(),
            }],
            projection: Projection::new(
                10,
                7,
                &(0..70)
                    .map(|i| (i as f32 - 35.0) / 64.0)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            bias: vec![0.0625; 7],
            input_width: 10,
        }
    }

    #[test]
    fn persistent_session_resets_state_reuses_output_and_matches_both_budgets() {
        for backend in [Backend::Scalar, Backend::Neon] {
            if backend == Backend::Neon && !Backend::neon_available() {
                continue;
            }
            let oracle = toy_model();
            let mut one = toy_model().into_session(backend, 1).unwrap();
            let mut two = toy_model().into_session(backend, 2).unwrap();
            assert_eq!(one.threads(), 1);
            assert_eq!(two.threads(), 2);
            let warmup = vec![0.25; 17 * 10];
            let one_pointer = one.infer(&warmup, 17).unwrap().as_ptr();
            let two_pointer = two.infer(&warmup, 17).unwrap().as_ptr();
            for rows in [3, 17, 5, 17, 1, 17] {
                let input: Vec<_> = (0..rows * 10)
                    .map(|i| ((i * 13 % 29) as f32 - 14.0) / 16.0)
                    .collect();
                let expected = oracle.infer(&input, rows, backend).unwrap();
                let actual = one.infer(&input, rows).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(actual.as_ptr(), one_pointer);
                let actual = two.infer(&input, rows).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(actual.as_ptr(), two_pointer);
            }
            assert!(two.infer(&[], 0).is_err());
            assert!(one.infer(&[f32::NAN; 10], 1).is_err());
            assert_eq!(
                one.infer(&warmup, 17).unwrap(),
                two.infer(&warmup, 17).unwrap()
            );
        }
    }

    #[test]
    fn inference_budget_rejects_zero_and_oversubscription() {
        for threads in [0, 3, 8, usize::MAX] {
            assert!(toy_model().into_session(Backend::Scalar, threads).is_err());
        }
    }

    #[test]
    fn reject_truncated_and_unrelated_models() {
        for bytes in [&[][..], b"TFL3", b"\x08\0\0\0TFL3", b"\xff\xff\xff\xffTFL3"] {
            assert!(Model::from_tflite(bytes).is_err());
        }
    }

    #[test]
    fn reverse_direction_keeps_original_frame_order_and_resets_state() {
        let units = 5;
        let d = Direction {
            projection: Projection::new(
                2,
                units * 4,
                &(0..units * 8)
                    .map(|i| (i as f32 - 20.0) / 32.0)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            recurrent: std::array::from_fn(|_| vec![0.25; units]),
            bias: std::array::from_fn(|_| vec![0.125; units]),
            units,
        };
        let x = [0.1, 0.2, -0.3, 0.4, 0.5, -0.6];
        let reversed: Vec<_> = x.chunks_exact(2).rev().flatten().copied().collect();
        let fw = d.run(&reversed, 3, false, Backend::Scalar);
        let expected: Vec<_> = fw.chunks_exact(units).rev().flatten().copied().collect();
        let actual = d.run(&x, 3, true, Backend::Scalar);
        assert_eq!(actual, expected);
        assert_eq!(actual, d.run(&x, 3, true, Backend::Scalar));
        if Backend::neon_available() {
            let simd = d.run(&x, 3, true, Backend::Neon);
            for (a, b) in actual.into_iter().zip(simd) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }
}
