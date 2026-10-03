//! Minimal NumPy `.npy` float32 matrix I/O for reference checks and benchmarks.

use crate::{error, Result};

pub fn read(path: impl AsRef<std::path::Path>) -> Result<(usize, usize, Vec<f32>)> {
    let bytes = std::fs::read(path)?;
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err(error("Expected a NumPy array"));
    }
    let version = *bytes.get(6).ok_or_else(|| error("Truncated NPY header"))?;
    let (start, size): (usize, usize) = match version {
        1 => (
            10,
            u16::from_le_bytes(
                bytes
                    .get(8..10)
                    .ok_or_else(|| error("Truncated NPY header"))?
                    .try_into()
                    .unwrap(),
            ) as usize,
        ),
        2 | 3 => (
            12,
            u32::from_le_bytes(
                bytes
                    .get(8..12)
                    .ok_or_else(|| error("Truncated NPY header"))?
                    .try_into()
                    .unwrap(),
            ) as usize,
        ),
        _ => return Err(error("Unsupported NPY version")),
    };
    let end = start
        .checked_add(size)
        .ok_or_else(|| error("NPY length overflow"))?;
    let header = std::str::from_utf8(
        bytes
            .get(start..end)
            .ok_or_else(|| error("Truncated NPY header"))?,
    )
    .map_err(|_| error("Invalid NPY header"))?;
    if !header.contains("'<f4'") || !header.contains("'fortran_order': False") {
        return Err(error("Expected row-major little-endian float32 NPY"));
    }
    let shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| error("Missing NPY shape"))?;
    let dims = shape
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| error("Invalid NPY shape"))
        })
        .collect::<Result<Vec<_>>>()?;
    if dims.len() != 2 {
        return Err(error("Expected a two-dimensional matrix"));
    }
    let size = dims[0]
        .checked_mul(dims[1])
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| error("NPY shape overflow"))?;
    let data = bytes.get(end..).ok_or_else(|| error("Missing NPY data"))?;
    if data.len() != size {
        return Err(error("NPY data length mismatch"));
    }
    Ok((
        dims[0],
        dims[1],
        data.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    ))
}

pub fn write(
    path: impl AsRef<std::path::Path>,
    rows: usize,
    columns: usize,
    data: &[f32],
) -> Result<()> {
    if rows.checked_mul(columns) != Some(data.len()) {
        return Err(error("Output matrix dimensions mismatch"));
    }
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}");
    let padding = (16 - (10 + header.len() + 1) % 16) % 16;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + 4 * data.len());
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}
