//! The alphabet in the recognition-spec protobuf, without generated bindings.

use crate::{error, Result};

fn varint(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| error("Truncated protobuf varint"))?;
        *cursor += 1;
        if shift == 63 && byte > 1 {
            return Err(error("Protobuf varint overflow"));
        }
        value |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return Ok(value);
        }
    }
    Err(error("Protobuf varint overflow"))
}

fn messages(bytes: &[u8], wanted: u32) -> Result<Vec<&[u8]>> {
    let mut cursor = 0;
    let mut matches = Vec::new();
    while cursor < bytes.len() {
        let key = varint(bytes, &mut cursor)?;
        if key >> 3 == 0 {
            return Err(error("Invalid protobuf field"));
        }
        let width = match key & 7 {
            0 => {
                varint(bytes, &mut cursor)?;
                continue;
            }
            1 => 8,
            2 => usize::try_from(varint(bytes, &mut cursor)?)
                .map_err(|_| error("Protobuf length overflow"))?,
            5 => 4,
            _ => return Err(error("Unsupported protobuf wire type")),
        };
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| error("Protobuf length overflow"))?;
        let field = bytes
            .get(cursor..end)
            .ok_or_else(|| error("Truncated protobuf field"))?;
        if key >> 3 == u64::from(wanted) && key & 7 == 2 {
            matches.push(field);
        }
        cursor = end;
    }
    Ok(matches)
}

pub fn alphabet(bytes: &[u8]) -> Result<Vec<String>> {
    let ext = messages(bytes, 158_518_157)?
        .into_iter()
        .next()
        .ok_or_else(|| error("Missing recognizer extension"))?;
    let alphabet = messages(ext, 2)?
        .into_iter()
        .next()
        .ok_or_else(|| error("Missing alphabet"))?;
    let chars = messages(alphabet, 1)?
        .into_iter()
        .map(|v| {
            String::from_utf8(v.to_vec()).map_err(|_| error("Alphabet contains invalid UTF-8"))
        })
        .collect::<Result<Vec<_>>>()?;
    if chars.is_empty() {
        return Err(error("Empty alphabet"));
    }
    Ok(chars)
}

pub fn decoder_weights(bytes: &[u8]) -> Result<(f64, f64)> {
    let ext = messages(bytes, 158_518_157)?
        .into_iter()
        .next()
        .ok_or_else(|| error("Missing recognizer extension"))?;
    let config = messages(ext, 4)?
        .into_iter()
        .next()
        .ok_or_else(|| error("Missing decoder configuration"))?;
    let fst = messages(config, 6)?
        .into_iter()
        .next()
        .ok_or_else(|| error("Missing FST decoder configuration"))?;
    let (mut cursor, mut weight, mut bonus) = (0, None, None);
    while cursor < fst.len() {
        let key = varint(fst, &mut cursor)?;
        let width = match key & 7 {
            0 => {
                varint(fst, &mut cursor)?;
                continue;
            }
            1 => 8,
            2 => varint(fst, &mut cursor)? as usize,
            5 => 4,
            _ => return Err(error("Unsupported decoder wire type")),
        };
        let value = fst
            .get(cursor..cursor + width)
            .ok_or_else(|| error("Truncated decoder configuration"))?;
        if key & 7 == 5 {
            let number = f32::from_le_bytes(value.try_into().unwrap()) as f64;
            if key >> 3 == 7 {
                weight = Some(number);
            }
            if key >> 3 == 10 {
                bonus = Some(number);
            }
        }
        cursor += width;
    }
    Ok((
        weight.ok_or_else(|| error("Missing LM weight"))?,
        bonus.ok_or_else(|| error("Missing character bonus"))?,
    ))
}

pub fn greedy(logits: &[f32], alphabet: &[String]) -> Result<String> {
    let columns = alphabet.len() + 1;
    if !logits.len().is_multiple_of(columns) {
        return Err(error("Logit dimensions do not match alphabet"));
    }
    let blank = alphabet.len();
    let mut last = blank;
    let mut text = String::new();
    for row in logits.chunks_exact(columns) {
        let mut best = 0;
        for i in 1..columns {
            if row[i] > row[best] {
                best = i;
            }
        }
        if best != blank && best != last {
            text.push_str(&alphabet[best]);
        }
        last = best;
    }
    Ok(text)
}
