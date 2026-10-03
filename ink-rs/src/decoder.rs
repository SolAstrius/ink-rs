//! Compact character LM and Viterbi CTC beam search matching the reference decoder.

use crate::{error, Result};
use std::collections::HashMap;

pub struct CompactLm {
    bytes: Vec<u8>,
    states: usize,
    start: usize,
    step: f64,
    context_zeros: Vec<u32>,
    future_zeros: Vec<u32>,
    final_offset: usize,
    final_rank: Vec<u32>,
    context_labels: usize,
    future_labels: usize,
    backoff_costs: usize,
    final_costs: usize,
    future_costs: usize,
    symbols: Vec<Option<u16>>,
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| error("Truncated FST"))?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| error("Truncated FST"))?
            .try_into()
            .unwrap(),
    ))
}
fn bitmap(bytes: &[u8], offset: &mut usize, bits: usize) -> Result<(usize, Vec<u32>)> {
    let start = *offset;
    let words = bits.div_ceil(64);
    let mut zeros = Vec::new();
    for i in 0..words {
        let word = u64_at(bytes, start + i * 8)?;
        let valid = (bits - i * 64).min(64);
        let mask = if valid == 64 {
            u64::MAX
        } else {
            (1u64 << valid) - 1
        };
        let mut remaining = !word & mask;
        while remaining != 0 {
            zeros.push((i * 64 + remaining.trailing_zeros() as usize) as u32);
            remaining &= remaining - 1;
        }
    }
    *offset += words * 8;
    Ok((start, zeros))
}

impl CompactLm {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if u32_at(&bytes, 0)? != 0x7eb2_fdd6 {
            return Err(error("Not an OpenFst file"));
        }
        let mut cursor = 4;
        let mut strings = Vec::new();
        for _ in 0..2 {
            let size = u32_at(&bytes, cursor)? as usize;
            cursor += 4;
            strings.push(
                bytes
                    .get(cursor..cursor + size)
                    .ok_or_else(|| error("Truncated FST header"))?
                    .to_vec(),
            );
            cursor += size;
        }
        if strings[0] != b"compact_lm" || strings[1] != b"standard" {
            return Err(error("Unsupported FST type"));
        }
        if u32_at(&bytes, cursor)? != 2 {
            return Err(error("Unsupported FST version"));
        }
        let start = u64_at(&bytes, cursor + 16)? as usize;
        cursor += 40;
        let symbol_count = u32_at(&bytes, cursor)? as usize;
        let step = f32::from_bits(u32_at(&bytes, cursor + 9)?) as f64;
        let states = u64_at(&bytes, 0x60)? as usize;
        let futures = u64_at(&bytes, 0x68)? as usize;
        let finals = u64_at(&bytes, 0x70)? as usize;
        if !(2..=10_000_000).contains(&states)
            || futures > 50_000_000
            || symbol_count > 65535
            || start >= states
            || !step.is_finite()
            || step <= 0.0
        {
            return Err(error("Invalid compact LM dimensions"));
        }
        let mut offset = 0x78;
        let (_, context_zeros) = bitmap(&bytes, &mut offset, 2 * states + 1)?;
        let (_, future_zeros) = bitmap(&bytes, &mut offset, states + futures + 1)?;
        let (final_offset, _) = bitmap(&bytes, &mut offset, states + 1)?;
        if context_zeros.len() != states + 1 || future_zeros.len() != states + 1 {
            return Err(error("Invalid LM tree bitmaps"));
        }
        let mut final_rank = Vec::with_capacity((states + 1).div_ceil(64) + 1);
        final_rank.push(0);
        for i in 0..(states + 1).div_ceil(64) {
            final_rank.push(
                final_rank.last().unwrap() + u64_at(&bytes, final_offset + i * 8)?.count_ones(),
            );
        }
        let context_labels = offset;
        offset += 2 * states + 2;
        let future_labels = offset;
        offset += 2 * futures;
        let backoff_costs = offset;
        offset += states + 1;
        let final_costs = offset;
        offset += finals;
        let future_costs = offset;
        offset += futures;
        let symbol_bitmap = (offset + 3) & !3;
        let bitmap_size = symbol_count.div_ceil(32) * 4;
        if symbol_bitmap + bitmap_size > bytes.len() {
            return Err(error("Truncated LM symbol bitmap"));
        }
        let mut symbols = Vec::with_capacity(symbol_count);
        let mut label = 0;
        for i in 0..symbol_count {
            if bytes[symbol_bitmap + i / 8] & (1 << (i % 8)) != 0 {
                symbols.push(Some(label));
                label += 1;
            } else {
                symbols.push(None);
            }
        }
        Ok(Self {
            bytes,
            states,
            start,
            step,
            context_zeros,
            future_zeros,
            final_offset,
            final_rank,
            context_labels,
            future_labels,
            backoff_costs,
            final_costs,
            future_costs,
            symbols,
        })
    }

    fn label_at(&self, offset: usize, index: usize) -> u16 {
        let start = offset + index * 2;
        u16::from_le_bytes([self.bytes[start], self.bytes[start + 1]])
    }
    fn cost(&self, byte: u8) -> f64 {
        if byte >= 254 {
            f64::INFINITY
        } else {
            byte as f64 * self.step
        }
    }
    fn first_child(&self, node: usize) -> usize {
        self.context_zeros[node] as usize - node
    }
    fn parent(&self, node: usize) -> usize {
        let (mut left, mut right) = (0, self.states);
        while left < right {
            let mid = (left + right) / 2;
            if self.first_child(mid) <= node {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left.saturating_sub(1)
    }
    fn find_label(&self, offset: usize, start: usize, count: usize, label: u16) -> Option<usize> {
        let (mut low, mut high) = (0, count);
        while low < high {
            let mid = (low + high) / 2;
            if self.label_at(offset, start + mid) < label {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        (low < count && self.label_at(offset, start + low) == label).then_some(start + low)
    }
    fn child(&self, node: usize, label: u16) -> Option<usize> {
        let start = self.first_child(node);
        let count = (self.context_zeros[node + 1] - self.context_zeros[node] - 1) as usize;
        self.find_label(self.context_labels, start, count, label)
    }
    fn next_state(&self, node: usize, label: u16) -> usize {
        let mut history = [0u16; 64];
        let mut length = 0;
        let mut previous = node;
        while previous != 0 {
            if length == history.len() {
                return 0;
            }
            history[length] = self.label_at(self.context_labels, previous);
            length += 1;
            previous = self.parent(previous);
        }
        let Some(mut current) = self.child(0, label) else {
            return 0;
        };
        for &label in history[..length].iter().rev() {
            let Some(next) = self.child(current, label) else {
                break;
            };
            current = next;
        }
        current
    }
    fn score_label(&self, node: usize, label: u16) -> (f64, usize) {
        let (mut current, mut total) = (node, 0.0);
        loop {
            let start = self.future_zeros[current] as usize - current;
            let count = (self.future_zeros[current + 1] - self.future_zeros[current] - 1) as usize;
            if let Some(index) = self.find_label(self.future_labels, start, count, label) {
                return (
                    total + self.cost(self.bytes[self.future_costs + index]),
                    self.next_state(node, label),
                );
            }
            if current == 0 {
                return (f64::INFINITY, 0);
            }
            total += self.cost(self.bytes[self.backoff_costs + current]);
            current = self.parent(current);
        }
    }
    fn final_cost(&self, node: usize) -> f64 {
        let (mut current, mut total) = (node, 0.0);
        loop {
            let word_index = current / 64;
            let bit_index = current % 64;
            let word = u64_at(&self.bytes, self.final_offset + word_index * 8).unwrap();
            if word & (1u64 << bit_index) != 0 {
                let before = if bit_index == 0 {
                    0
                } else {
                    word & ((1u64 << bit_index) - 1)
                };
                let rank = self.final_rank[word_index] as usize + before.count_ones() as usize;
                return total + self.cost(self.bytes[self.final_costs + rank]);
            }
            if current == 0 {
                return f64::INFINITY;
            }
            total += self.cost(self.bytes[self.backoff_costs + current]);
            current = self.parent(current);
        }
    }
}

#[derive(Clone, Copy)]
pub struct SearchOptions {
    pub beam: f64,
    pub max_active: usize,
    pub nbest: usize,
    pub lm_weight: f64,
    pub char_bonus: f64,
}

pub struct Hypothesis {
    pub text: String,
    pub cost: f64,
}

#[derive(Clone, Copy)]
struct Prefix {
    blank: f64,
    nonblank: f64,
    state: usize,
}

pub struct BeamDecoder {
    lm: CompactLm,
    alphabet: Vec<String>,
    labels: Vec<Option<u16>>,
    cache: HashMap<(usize, u16), (f64, usize)>,
}

impl BeamDecoder {
    pub fn new(lm: CompactLm, alphabet: Vec<String>) -> Result<Self> {
        if alphabet.len() + 2 > lm.symbols.len() {
            return Err(error("LM/alphabet dimensions differ"));
        }
        let labels = (0..alphabet.len()).map(|i| lm.symbols[i + 2]).collect();
        Ok(Self {
            lm,
            alphabet,
            labels,
            cache: HashMap::new(),
        })
    }
    fn lm_step(&mut self, state: usize, token: usize) -> (f64, usize) {
        let Some(label) = self.labels[token] else {
            return (f64::INFINITY, 0);
        };
        if let Some(value) = self.cache.get(&(state, label)) {
            return *value;
        }
        let value = self.lm.score_label(state, label);
        if self.cache.len() >= 100_000 {
            self.cache.clear();
        }
        self.cache.insert((state, label), value);
        value
    }

    pub fn decode(&mut self, logits: &[f32], options: SearchOptions) -> Result<Vec<Hypothesis>> {
        let blank = self.alphabet.len();
        let columns = blank + 1;
        if logits.is_empty()
            || !logits.len().is_multiple_of(columns)
            || logits.iter().any(|v| !v.is_finite())
        {
            return Err(error("Invalid decoder logits"));
        }
        if options.beam <= 0.0
            || !options.beam.is_finite()
            || options.max_active == 0
            || options.nbest == 0
        {
            return Err(error(format!(
                "Invalid search options: beam={}, max_active={}, nbest={}",
                options.beam, options.max_active, options.nbest
            )));
        }
        let mut prefixes = vec![(
            Vec::<u16>::new(),
            Prefix {
                blank: 0.0,
                nonblank: f64::INFINITY,
                state: self.lm.start,
            },
        )];
        for row in logits.chunks_exact(columns) {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let candidates: Vec<_> = (0..blank)
                .filter(|i| row[*i] as f64 >= maximum - options.beam)
                .collect();
            let mut next = HashMap::<Vec<u16>, Prefix>::new();
            for (text, prefix) in prefixes {
                let best = prefix.blank.min(prefix.nonblank);
                let sustain = text.last().map_or(f64::INFINITY, |last| {
                    prefix.nonblank - row[*last as usize] as f64
                });
                let entry = next.entry(text.clone()).or_insert(Prefix {
                    blank: f64::INFINITY,
                    nonblank: f64::INFINITY,
                    state: prefix.state,
                });
                entry.blank = entry.blank.min(best - row[blank] as f64);
                entry.nonblank = entry.nonblank.min(sustain);
                for &token in &candidates {
                    let source = if text.last().copied() == Some(token as u16) {
                        prefix.blank
                    } else {
                        best
                    };
                    if !source.is_finite() {
                        continue;
                    }
                    let (cost, state) = self.lm_step(prefix.state, token);
                    if !cost.is_finite() {
                        continue;
                    }
                    let score =
                        source - row[token] as f64 + options.lm_weight * cost + options.char_bonus;
                    let mut extended = text.clone();
                    extended.push(token as u16);
                    let entry = next.entry(extended).or_insert(Prefix {
                        blank: f64::INFINITY,
                        nonblank: f64::INFINITY,
                        state,
                    });
                    entry.nonblank = entry.nonblank.min(score);
                }
            }
            prefixes = next.into_iter().collect();
            prefixes.sort_by(|a, b| {
                a.1.blank
                    .min(a.1.nonblank)
                    .total_cmp(&b.1.blank.min(b.1.nonblank))
                    .then_with(|| a.0.cmp(&b.0))
            });
            if prefixes.is_empty() {
                return Err(error("Decoder search ran out of hypotheses"));
            }
            let best = prefixes[0].1.blank.min(prefixes[0].1.nonblank);
            prefixes.truncate(options.max_active);
            prefixes.retain(|(_, p)| p.blank.min(p.nonblank) <= best + options.beam);
        }
        let mut results: Vec<_> = prefixes
            .into_iter()
            .map(|(tokens, p)| Hypothesis {
                text: tokens
                    .into_iter()
                    .map(|i| self.alphabet[i as usize].as_str())
                    .collect(),
                cost: p.blank.min(p.nonblank) + options.lm_weight * self.lm.final_cost(p.state),
            })
            .collect();
        results.sort_by(|a, b| a.cost.total_cmp(&b.cost).then_with(|| a.text.cmp(&b.text)));
        results.truncate(options.nbest);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_fst_is_rejected() {
        for data in [vec![], vec![0; 100], 0x7eb2_fdd6_u32.to_le_bytes().to_vec()] {
            assert!(CompactLm::from_bytes(data).is_err());
        }
    }
}
