use crate::{error, Result};

/// Scalar is portable; Auto selects NEON when the running CPU supports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Scalar,
    Neon,
}

impl Backend {
    pub fn resolve(self) -> Result<Self> {
        match self {
            Self::Auto => Ok(if Self::neon_available() {
                Self::Neon
            } else {
                Self::Scalar
            }),
            Self::Neon if !Self::neon_available() => Err(error("NEON is unavailable on this CPU")),
            backend => Ok(backend),
        }
    }

    pub fn neon_available() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            return std::arch::is_aarch64_feature_detected!("neon");
        }
        #[cfg(all(target_arch = "arm", target_os = "linux"))]
        {
            // AT_HWCAP in the process's native 32-bit ELF auxiliary vector.
            // HWCAP_NEON is bit 12. This avoids unstable ARM feature macros.
            if let Ok(auxv) = std::fs::read("/proc/self/auxv") {
                for entry in auxv.chunks_exact(8) {
                    let key = u32::from_ne_bytes(entry[..4].try_into().unwrap());
                    let value = u32::from_ne_bytes(entry[4..].try_into().unwrap());
                    if key == 16 {
                        return value & (1 << 12) != 0;
                    }
                }
            }
        }
        #[allow(unreachable_code)]
        {
            false
        }
    }
}

/// An immutable packed input can be shared by the two directions of a layer.
pub(crate) struct PreparedInput {
    pub rows: usize,
    pub columns: usize,
    pub backend: Backend,
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    time_tile: usize,
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    padded_rows: usize,
    data: Vec<f32>,
}

impl Default for PreparedInput {
    fn default() -> Self {
        Self {
            rows: 0,
            columns: 0,
            backend: Backend::Scalar,
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            time_tile: 4,
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            padded_rows: 0,
            data: Vec::new(),
        }
    }
}

impl PreparedInput {
    /// Merge direction outputs directly into the next packed input, avoiding
    /// a row-major concatenation buffer and a second copy of the same values.
    pub fn prepare_directions(
        &mut self,
        forward: &[f32],
        backward: &[f32],
        rows: usize,
        units: usize,
        backend: Backend,
    ) {
        debug_assert_eq!(forward.len(), rows * units);
        debug_assert_eq!(backward.len(), rows * units);
        self.rows = rows;
        self.columns = units * 2;
        self.backend = backend;
        if backend == Backend::Scalar {
            self.data.resize(rows * units * 2, 0.0);
            for row in 0..rows {
                self.data[row * units * 2..row * units * 2 + units]
                    .copy_from_slice(&forward[row * units..(row + 1) * units]);
                self.data[row * units * 2 + units..(row + 1) * units * 2]
                    .copy_from_slice(&backward[row * units..(row + 1) * units]);
            }
            return;
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            self.time_tile = if cfg!(target_arch = "aarch64") {
                if rows <= 4 {
                    4
                } else if rows.div_ceil(12) * 12 <= rows.div_ceil(8) * 8 {
                    12
                } else {
                    8
                }
            } else {
                4
            };
            self.padded_rows = rows.div_ceil(self.time_tile) * self.time_tile;
            self.data.resize(self.padded_rows * self.columns, 0.0);
            #[cfg(target_arch = "aarch64")]
            for block in (0..self.padded_rows).step_by(self.time_tile) {
                for k in 0..self.columns {
                    let (source, column) = if k < units {
                        (forward, k)
                    } else {
                        (backward, k - units)
                    };
                    for lane in 0..self.time_tile {
                        self.data[block * self.columns + k * self.time_tile + lane] =
                            if block + lane < rows {
                                source[(block + lane) * units + column]
                            } else {
                                0.0
                            };
                    }
                }
            }
            #[cfg(target_arch = "arm")]
            {
                for row in 0..rows {
                    self.data[row * units * 2..row * units * 2 + units]
                        .copy_from_slice(&forward[row * units..(row + 1) * units]);
                    self.data[row * units * 2 + units..(row + 1) * units * 2]
                        .copy_from_slice(&backward[row * units..(row + 1) * units]);
                }
                self.data[rows * self.columns..].fill(0.0);
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
        unreachable!("NEON cannot be selected on this architecture");
    }

    pub fn prepare(&mut self, input: &[f32], rows: usize, columns: usize, backend: Backend) {
        debug_assert_eq!(input.len(), rows * columns);
        self.rows = rows;
        self.columns = columns;
        self.backend = backend;
        if backend == Backend::Scalar {
            self.data.resize(input.len(), 0.0);
            self.data.copy_from_slice(input);
            return;
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            self.time_tile = if cfg!(target_arch = "aarch64") {
                if rows <= 4 {
                    4
                } else if rows.div_ceil(12) * 12 <= rows.div_ceil(8) * 8 {
                    12
                } else {
                    8
                }
            } else {
                4
            };
            self.padded_rows = rows.div_ceil(self.time_tile) * self.time_tile;
            self.data.resize(self.padded_rows * columns, 0.0);
            #[cfg(target_arch = "aarch64")]
            for block in (0..self.padded_rows).step_by(self.time_tile) {
                for k in 0..columns {
                    for lane in 0..self.time_tile {
                        self.data[block * columns + k * self.time_tile + lane] =
                            if block + lane < rows {
                                input[(block + lane) * columns + k]
                            } else {
                                0.0
                            };
                    }
                }
            }
            #[cfg(target_arch = "arm")]
            {
                self.data[..input.len()].copy_from_slice(input);
                self.data[input.len()..].fill(0.0);
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
        unreachable!("NEON cannot be selected on this architecture");
    }
}

pub(crate) struct Projection {
    pub inputs: usize,
    pub outputs: usize,
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    stride: usize,
    // Eight-output panels: [output_panel][input][lane].
    weights: Vec<f32>,
}

impl Projection {
    pub fn new(inputs: usize, outputs: usize, row_major: &[f32]) -> Result<Self> {
        if inputs == 0 || outputs == 0 || inputs.checked_mul(outputs) != Some(row_major.len()) {
            return Err(error("Invalid projection dimensions"));
        }
        let stride = outputs
            .checked_add(7)
            .ok_or_else(|| error("Projection overflow"))?
            / 8
            * 8;
        let size = inputs
            .checked_mul(stride)
            .ok_or_else(|| error("Projection overflow"))?;
        let mut weights = vec![0.0; size];
        for output in 0..outputs {
            for input in 0..inputs {
                weights[(output / 8) * inputs * 8 + input * 8 + output % 8] =
                    row_major[output * inputs + input];
            }
        }
        Ok(Self {
            inputs,
            outputs,
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            stride,
            weights,
        })
    }

    /// Reuses output capacity and never reads values left by a previous call.
    pub fn forward_into(&self, prepared: &PreparedInput, output: &mut Vec<f32>) {
        debug_assert_eq!(prepared.columns, self.inputs);
        if prepared.backend == Backend::Scalar {
            output.resize(prepared.rows * self.outputs, 0.0);
            for t in 0..prepared.rows {
                for j in 0..self.outputs {
                    let mut sum = 0.0;
                    for k in 0..self.inputs {
                        sum += prepared.data[t * self.inputs + k]
                            * self.weights[(j / 8) * self.inputs * 8 + k * 8 + j % 8];
                    }
                    output[t * self.outputs + j] = sum;
                }
            }
            return;
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            output.resize(prepared.padded_rows * self.stride, 0.0);
            for column in (0..self.stride).step_by(8) {
                for row in (0..prepared.padded_rows).step_by(prepared.time_tile) {
                    // The checked/padded prepared input provides all rows and
                    // lanes. Each kernel overwrites its complete output tile.
                    unsafe {
                        #[cfg(target_arch = "aarch64")]
                        match prepared.time_tile {
                            4 => neon::tile::<4>(
                                self.weights.as_ptr().add(column * self.inputs),
                                prepared.data.as_ptr().add(row * self.inputs),
                                output.as_mut_ptr().add(row * self.stride + column),
                                self.inputs,
                                self.stride,
                            ),
                            8 => neon::tile::<8>(
                                self.weights.as_ptr().add(column * self.inputs),
                                prepared.data.as_ptr().add(row * self.inputs),
                                output.as_mut_ptr().add(row * self.stride + column),
                                self.inputs,
                                self.stride,
                            ),
                            12 => neon::tile::<12>(
                                self.weights.as_ptr().add(column * self.inputs),
                                prepared.data.as_ptr().add(row * self.inputs),
                                output.as_mut_ptr().add(row * self.stride + column),
                                self.inputs,
                                self.stride,
                            ),
                            _ => unreachable!(),
                        }
                        #[cfg(target_arch = "arm")]
                        neon::tile4x8(
                            self.weights.as_ptr().add(column * self.inputs),
                            prepared.data.as_ptr().add(row * self.inputs),
                            output.as_mut_ptr().add(row * self.stride + column),
                            self.inputs,
                            self.stride,
                        );
                    }
                }
            }
            // Compact channel padding in place, preserving allocation capacity.
            if self.outputs != self.stride {
                for row in 0..prepared.rows {
                    output.copy_within(
                        row * self.stride..row * self.stride + self.outputs,
                        row * self.outputs,
                    );
                }
            }
            output.truncate(prepared.rows * self.outputs);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
        unreachable!("NEON cannot be selected on this architecture");
    }

    #[cfg(test)]
    pub fn forward(&self, input: &[f32], rows: usize, backend: Backend) -> Vec<f32> {
        let mut prepared = PreparedInput::default();
        prepared.prepare(input, rows, self.inputs, backend);
        let mut output = Vec::new();
        self.forward_into(&prepared, &mut output);
        output
    }
}

#[inline]
pub(crate) fn affine4(
    pre: &[f32],
    bias: &[f32],
    recurrent: &[f32],
    hidden: &[f32],
    out: &mut [f32; 4],
) {
    debug_assert!(pre.len() >= 4 && bias.len() >= 4 && recurrent.len() >= 4 && hidden.len() >= 4);
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        neon::affine4(
            pre.as_ptr(),
            bias.as_ptr(),
            recurrent.as_ptr(),
            hidden.as_ptr(),
            out.as_mut_ptr(),
        );
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
    for j in 0..4 {
        out[j] = (pre[j] + bias[j]) + recurrent[j] * hidden[j];
    }
}

#[inline]
pub(crate) fn cell4(
    forget: &[f32; 4],
    previous: &[f32],
    input: &[f32; 4],
    candidate: &[f32; 4],
    out: &mut [f32; 4],
) {
    debug_assert!(previous.len() >= 4);
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        neon::cell4(
            forget.as_ptr(),
            previous.as_ptr(),
            input.as_ptr(),
            candidate.as_ptr(),
            out.as_mut_ptr(),
        );
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
    for j in 0..4 {
        out[j] = forget[j] * previous[j] + input[j] * candidate[j];
    }
}

#[inline]
pub(crate) fn mul4(left: &[f32; 4], right: &[f32; 4], out: &mut [f32; 4]) {
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        neon::mul4(left.as_ptr(), right.as_ptr(), out.as_mut_ptr());
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
    for j in 0..4 {
        out[j] = left[j] * right[j];
    }
}

#[inline]
pub(crate) fn activation4(values: &mut [f32; 4], tanh: bool) {
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        neon::activation4(values.as_mut_ptr(), tanh);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
    for value in values {
        *value = if tanh {
            value.tanh()
        } else {
            1.0 / (1.0 + (-*value).exp())
        };
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    // exp(x) = 2^n * exp(r), |r| <= ln(2)/2. A degree-seven Taylor
    // polynomial is sufficient for float32 here. Split ln(2) retains
    // accuracy in range reduction, including the non-fused ARMv7 path.
    #[inline(always)]
    unsafe fn exp_negative(x: float32x4_t) -> float32x4_t {
        unsafe {
            let floor = vdupq_n_f32(-87.0);
            let clamped = vmaxq_f32(x, floor);
            let n = vcvtnq_s32_f32(vmulq_n_f32(clamped, std::f32::consts::LOG2_E));
            let nf = vcvtq_f32_s32(n);
            let mut r = vfmaq_n_f32(clamped, nf, -f32::from_bits(0x3f317200));
            r = vfmaq_n_f32(r, nf, -1.428_606_8e-6);
            let mut p = vdupq_n_f32(1.0 / 5040.0);
            for coefficient in [
                1.0 / 720.0,
                1.0 / 120.0,
                1.0 / 24.0,
                1.0 / 6.0,
                0.5,
                1.0,
                1.0,
            ] {
                p = vfmaq_f32(vdupq_n_f32(coefficient), p, r);
            }
            let exponent = vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127)));
            let result = vmulq_f32(p, vreinterpretq_f32_s32(exponent));
            vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(result),
                vcgeq_f32(x, floor),
            ))
        }
    }

    #[inline(always)]
    pub unsafe fn activation4(values: *mut f32, tanh: bool) {
        unsafe {
            let x = vld1q_f32(values);
            let a = vabsq_f32(x);
            let one = vdupq_n_f32(1.0);
            let e = exp_negative(vmulq_n_f32(a, if tanh { -2.0 } else { -1.0 }));
            let inverse = vdivq_f32(one, vaddq_f32(one, e));
            let result = if tanh {
                let large = vmulq_f32(vsubq_f32(one, e), inverse);
                // Avoid cancellation near zero with the odd Taylor series.
                let small_mask = vcltq_f32(a, vdupq_n_f32(0.0625));
                let small_a = vbslq_f32(small_mask, a, vdupq_n_f32(0.0));
                let a2 = vmulq_f32(small_a, small_a);
                let mut p = vdupq_n_f32(62.0 / 2835.0);
                for coefficient in [-17.0 / 315.0, 2.0 / 15.0, -1.0 / 3.0] {
                    p = vfmaq_f32(vdupq_n_f32(coefficient), p, a2);
                }
                let small = vmulq_f32(small_a, vfmaq_f32(one, a2, p));
                let magnitude = vbslq_f32(small_mask, small, large);
                let sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000));
                vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(magnitude), sign))
            } else {
                vbslq_f32(
                    vcgeq_f32(x, vdupq_n_f32(0.0)),
                    inverse,
                    vmulq_f32(e, inverse),
                )
            };
            vst1q_f32(values, result);
        }
    }

    pub unsafe fn tile<const ROWS: usize>(
        weights: *const f32,
        input: *const f32,
        output: *mut f32,
        k: usize,
        stride: usize,
    ) {
        unsafe {
            let z = vdupq_n_f32(0.0);
            let mut lo = [z; ROWS];
            let mut hi = [z; ROWS];
            for i in 0..k {
                let w0 = vld1q_f32(weights.add(i * 8));
                let w1 = vld1q_f32(weights.add(i * 8 + 4));
                let x0 = vld1q_f32(input.add(i * ROWS));
                let x1 = if ROWS >= 8 {
                    vld1q_f32(input.add(i * ROWS + 4))
                } else {
                    z
                };
                let x2 = if ROWS >= 12 {
                    vld1q_f32(input.add(i * ROWS + 8))
                } else {
                    z
                };
                macro_rules! update {
                    ($row:literal, $x:ident, $lane:literal) => {
                        lo[$row] = vfmaq_laneq_f32::<$lane>(lo[$row], w0, $x);
                        hi[$row] = vfmaq_laneq_f32::<$lane>(hi[$row], w1, $x);
                    };
                }
                update!(0, x0, 0);
                update!(1, x0, 1);
                update!(2, x0, 2);
                update!(3, x0, 3);
                if ROWS >= 8 {
                    update!(4, x1, 0);
                    update!(5, x1, 1);
                    update!(6, x1, 2);
                    update!(7, x1, 3);
                }
                if ROWS >= 12 {
                    update!(8, x2, 0);
                    update!(9, x2, 1);
                    update!(10, x2, 2);
                    update!(11, x2, 3);
                }
            }
            for row in 0..ROWS {
                vst1q_f32(output.add(row * stride), lo[row]);
                vst1q_f32(output.add(row * stride + 4), hi[row]);
            }
        }
    }

    #[inline]
    pub unsafe fn affine4(
        pre: *const f32,
        bias: *const f32,
        recurrent: *const f32,
        hidden: *const f32,
        out: *mut f32,
    ) {
        unsafe {
            let p = vaddq_f32(vld1q_f32(pre), vld1q_f32(bias));
            let r = vmulq_f32(vld1q_f32(recurrent), vld1q_f32(hidden));
            vst1q_f32(out, vaddq_f32(p, r));
        }
    }

    #[inline]
    pub unsafe fn cell4(
        f: *const f32,
        previous: *const f32,
        i: *const f32,
        g: *const f32,
        out: *mut f32,
    ) {
        unsafe {
            let retained = vmulq_f32(vld1q_f32(f), vld1q_f32(previous));
            let added = vmulq_f32(vld1q_f32(i), vld1q_f32(g));
            vst1q_f32(out, vaddq_f32(retained, added));
        }
    }

    #[inline]
    pub unsafe fn mul4(a: *const f32, b: *const f32, out: *mut f32) {
        unsafe {
            vst1q_f32(out, vmulq_f32(vld1q_f32(a), vld1q_f32(b)));
        }
    }
}

#[cfg(target_arch = "arm")]
mod neon {
    use core::arch::asm;

    static ACTIVATION_CONSTANTS: [f32; 19] = [
        -87.0,
        std::f32::consts::LOG2_E,
        0.5,
        f32::from_bits(0x3f317200),
        1.428_606_8e-6,
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
        0.0625,
        62.0 / 2835.0,
        -17.0 / 315.0,
        2.0 / 15.0,
        -1.0 / 3.0,
        1.0,
    ];

    #[inline(always)]
    pub unsafe fn activation4(values: *mut f32, tanh: bool) {
        unsafe {
            if tanh {
                activation_impl::<1>(values);
            } else {
                activation_impl::<0>(values);
            }
        }
    }

    #[inline(always)]
    unsafe fn activation_impl<const TANH: usize>(values: *mut f32) {
        unsafe {
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{values}]",
                "vabs.f32 q1, q0",
                "vneg.f32 q2, q1",
                ".if {tanh}", "vadd.f32 q2, q2, q2", ".endif",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!", // floor
                "vcge.f32 q14, q2, q6",
                "vmax.f32 q2, q2, q6",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!", // log2(e)
                "vmul.f32 q3, q2, q6",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!", // 0.5
                "vadd.f32 q3, q3, q6",
                "vcvt.s32.f32 q4, q3",
                "vcvt.f32.s32 q5, q4",
                "vclt.f32 q7, q3, q5",
                "vshr.u32 q7, q7, #31",
                "vsub.i32 q4, q4, q7",
                "vcvt.f32.s32 q3, q4",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!", // ln2 hi
                "vmul.f32 q5, q3, q6", "vsub.f32 q2, q2, q5",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!", // ln2 lo
                "vmul.f32 q5, q3, q6", "vsub.f32 q2, q2, q5",
                "vmov.i32 q7, #127",
                "vadd.i32 q4, q4, q7", "vshl.i32 q4, q4, #23",
                "vld1.32 {{d10[], d11[]}}, [{constants}]!", // c7
                ".rept 7",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!",
                "vmul.f32 q5, q5, q2", "vadd.f32 q5, q5, q6",
                ".endr",
                "vmul.f32 q8, q5, q4", "vand q8, q8, q14",
                "vadd.f32 q9, q8, q6", // q6 retains 1.0
                "vrecpe.f32 q10, q9",
                "vrecps.f32 q11, q9, q10", "vmul.f32 q10, q10, q11",
                "vrecps.f32 q11, q9, q10", "vmul.f32 q10, q10, q11",
                ".if {tanh}",
                "vsub.f32 q5, q6, q8", "vmul.f32 q12, q5, q10",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!", // small limit
                "vclt.f32 q14, q1, q6", "vand q1, q1, q14",
                "vmul.f32 q2, q1, q1",
                "vld1.32 {{d10[], d11[]}}, [{constants}]!", // tanh c4
                ".rept 3",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!",
                "vmul.f32 q5, q5, q2", "vadd.f32 q5, q5, q6",
                ".endr",
                "vld1.32 {{d12[], d13[]}}, [{constants}]!", // 1.0
                "vmul.f32 q5, q5, q2", "vadd.f32 q5, q5, q6",
                "vmul.f32 q11, q1, q5", "vbsl q14, q11, q12",
                "vmov.i32 q3, #0x80000000", "vand q3, q0, q3",
                "vorr q14, q14, q3", "vst1.32 {{q14}}, [{values}]",
                ".else",
                "veor q13, q13, q13", "vcge.f32 q7, q0, q13",
                "vmul.f32 q8, q8, q10", "vbsl q7, q10, q8",
                "vst1.32 {{q7}}, [{values}]",
                ".endif",
                values = in(reg) values,
                constants = inout(reg) ACTIVATION_CONSTANTS.as_ptr() => _,
                tanh = const TANH,
                // Explicit D registers are required for this target: Q-only
                // clobbers did not make LLVM save the callee-saved D8-D15 bank
                // when NEON is enabled locally inside the assembly block.
                out("d0") _, out("d1") _, out("d2") _, out("d3") _,
                out("d4") _, out("d5") _, out("d6") _, out("d7") _,
                out("d8") _, out("d9") _, out("d10") _, out("d11") _,
                out("d12") _, out("d13") _, out("d14") _, out("d15") _,
                out("d16") _, out("d17") _, out("d18") _, out("d19") _,
                out("d20") _, out("d21") _, out("d22") _, out("d23") _,
                out("d24") _, out("d25") _, out("d26") _, out("d27") _,
                out("d28") _, out("d29") _,
                options(nostack),
            );
        }
    }

    // ARMv7's NEON intrinsics require unstable language features. Stable inline
    // assembly keeps this crate on stable Rust and uses baseline NEON vmla.
    // Selection is gated by Linux HWCAP_NEON. Each assembly block enables
    // NEON parsing locally, without a global unstable +neon compiler flag.
    pub unsafe fn tile4x8(
        weights: *const f32,
        input: *const f32,
        output: *mut f32,
        k: usize,
        stride: usize,
    ) {
        unsafe {
            asm!(
                ".fpu neon",
                "veor q0, q0, q0", "veor q1, q1, q1",
                "veor q2, q2, q2", "veor q3, q3, q3",
                "veor q4, q4, q4", "veor q5, q5, q5",
                "veor q6, q6, q6", "veor q7, q7, q7",
                "2:",
                "vld1.32 {{d16, d17, d18, d19}}, [{w}]",
                "vld1.32 {{d20[], d21[]}}, [{x0}]!",
                "vmla.f32 q0, q8, q10", "vmla.f32 q1, q9, q10",
                "vld1.32 {{d20[], d21[]}}, [{x1}]!",
                "vmla.f32 q2, q8, q10", "vmla.f32 q3, q9, q10",
                "vld1.32 {{d20[], d21[]}}, [{x2}]!",
                "vmla.f32 q4, q8, q10", "vmla.f32 q5, q9, q10",
                "vld1.32 {{d20[], d21[]}}, [{x3}]!",
                "vmla.f32 q6, q8, q10", "vmla.f32 q7, q9, q10",
                "add {w}, {w}, #32", "subs {n}, {n}, #1", "bne 2b",
                "vst1.32 {{d0, d1, d2, d3}}, [{y}]", "add {y}, {y}, {bytes}",
                "vst1.32 {{d4, d5, d6, d7}}, [{y}]", "add {y}, {y}, {bytes}",
                "vst1.32 {{d8, d9, d10, d11}}, [{y}]", "add {y}, {y}, {bytes}",
                "vst1.32 {{d12, d13, d14, d15}}, [{y}]",
                w = inout(reg) weights => _,
                x0 = inout(reg) input => _,
                x1 = inout(reg) input.add(k) => _,
                x2 = inout(reg) input.add(2 * k) => _,
                x3 = inout(reg) input.add(3 * k) => _,
                y = inout(reg) output => _,
                n = inout(reg) k => _,
                bytes = in(reg) stride * 4,
                out("d0") _, out("d1") _, out("d2") _, out("d3") _,
                out("d4") _, out("d5") _, out("d6") _, out("d7") _,
                out("d8") _, out("d9") _, out("d10") _, out("d11") _,
                out("d12") _, out("d13") _, out("d14") _, out("d15") _,
                out("d16") _, out("d17") _, out("d18") _, out("d19") _,
                out("d20") _, out("d21") _,
                options(nostack),
            );
        }
    }

    #[inline]
    pub unsafe fn affine4(
        pre: *const f32,
        bias: *const f32,
        recurrent: *const f32,
        hidden: *const f32,
        out: *mut f32,
    ) {
        unsafe {
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{h}]", "vld1.32 {{q1}}, [{u}]",
                "vld1.32 {{q2}}, [{p}]", "vld1.32 {{q3}}, [{b}]",
                "vmul.f32 q0, q0, q1", "vadd.f32 q2, q2, q3",
                "vadd.f32 q0, q0, q2", "vst1.32 {{q0}}, [{o}]",
                h = in(reg) hidden, u = in(reg) recurrent,
                p = in(reg) pre, b = in(reg) bias, o = in(reg) out,
                out("d0") _, out("d1") _, out("d2") _, out("d3") _,
                out("d4") _, out("d5") _, out("d6") _, out("d7") _,
                options(nostack),
            );
        }
    }

    #[inline]
    pub unsafe fn cell4(
        f: *const f32,
        previous: *const f32,
        i: *const f32,
        g: *const f32,
        out: *mut f32,
    ) {
        unsafe {
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{f}]", "vld1.32 {{q1}}, [{c}]",
                "vld1.32 {{q2}}, [{i}]", "vld1.32 {{q3}}, [{g}]",
                "vmul.f32 q0, q0, q1", "vmul.f32 q2, q2, q3",
                "vadd.f32 q0, q0, q2", "vst1.32 {{q0}}, [{o}]",
                f = in(reg) f, c = in(reg) previous,
                i = in(reg) i, g = in(reg) g, o = in(reg) out,
                out("d0") _, out("d1") _, out("d2") _, out("d3") _,
                out("d4") _, out("d5") _, out("d6") _, out("d7") _,
                options(nostack),
            );
        }
    }

    #[inline]
    pub unsafe fn mul4(a: *const f32, b: *const f32, out: *mut f32) {
        unsafe {
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{a}]", "vld1.32 {{q1}}, [{b}]",
                "vmul.f32 q0, q0, q1", "vst1.32 {{q0}}, [{o}]",
                a = in(reg) a, b = in(reg) b, o = in(reg) out,
                out("d0") _, out("d1") _, out("d2") _, out("d3") _, options(nostack),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_direction_packing_matches_canonical_concatenation() {
        for backend in [Backend::Scalar, Backend::Neon] {
            if backend == Backend::Neon && !Backend::neon_available() {
                continue;
            }
            for rows in [1, 3, 9, 12, 17] {
                for units in [1, 5, 216, 280] {
                    let forward: Vec<_> = (0..rows * units).map(|i| i as f32 / 16.0).collect();
                    let backward: Vec<_> = (0..rows * units)
                        .map(|i| -(i as f32 + 1.0) / 32.0)
                        .collect();
                    let mut canonical = Vec::new();
                    for row in 0..rows {
                        canonical.extend_from_slice(&forward[row * units..(row + 1) * units]);
                        canonical.extend_from_slice(&backward[row * units..(row + 1) * units]);
                    }
                    let mut expected = PreparedInput::default();
                    expected.prepare(&canonical, rows, 2 * units, backend);
                    let mut actual = PreparedInput::default();
                    actual.prepare_directions(&forward, &backward, rows, units, backend);
                    assert_eq!(actual.data, expected.data);
                    assert_eq!(actual.rows, expected.rows);
                    assert_eq!(actual.columns, expected.columns);
                }
            }
        }
    }

    #[test]
    fn vector_activations_match_reference_across_saturation_and_zero() {
        if !Backend::neon_available() {
            return;
        }
        let mut cases: Vec<f32> = (-32768..=32768).map(|i| i as f32 / 1024.0).collect();
        cases.extend([
            -128.0,
            -87.0,
            -0.0,
            0.0,
            1e-12,
            -1e-12,
            87.0,
            128.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]);
        for tanh in [false, true] {
            for chunk in cases.chunks(4) {
                let mut actual = [0.0; 4];
                actual[..chunk.len()].copy_from_slice(chunk);
                activation4(&mut actual, tanh);
                for (value, input) in actual.into_iter().zip(chunk) {
                    let expected = if tanh {
                        input.tanh()
                    } else {
                        1.0 / (1.0 + (-input).exp())
                    };
                    assert!(
                        (value - expected).abs() <= 3e-7,
                        "tanh={tanh}, x={input}, actual={value}, expected={expected}"
                    );
                    assert!(value.is_finite());
                    if tanh && *input == 0.0 {
                        assert_eq!(value.to_bits(), input.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn neon_projection_matches_scalar_with_padding() {
        if !Backend::neon_available() {
            return;
        }
        // Exercise both time and output-channel tails, including tiny tensors.
        for rows in [1, 3, 4, 5, 9, 12, 13, 17] {
            for inputs in [1, 10, 37] {
                for outputs in [1, 4, 7, 8, 9, 247, 314] {
                    let x: Vec<_> = (0..rows * inputs)
                        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
                        .collect();
                    let w: Vec<_> = (0..inputs * outputs)
                        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
                        .collect();
                    let p = Projection::new(inputs, outputs, &w).unwrap();
                    let expected = p.forward(&x, rows, Backend::Scalar);
                    let actual = p.forward(&x, rows, Backend::Neon);
                    assert_eq!(actual.len(), expected.len());
                    for (a, b) in actual.iter().zip(expected) {
                        assert!(
                            (a - b).abs() <= 1e-5,
                            "{rows}x{inputs}x{outputs}: {a} != {b}"
                        );
                    }
                }
            }
        }
    }
}
