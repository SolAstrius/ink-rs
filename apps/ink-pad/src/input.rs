//! Passive raw evdev decoding, using complete SYN_REPORT frames as scribble-rs does.

#[derive(Clone, Copy, Default, Debug)]
pub struct Report {
    pub time_ms: f64,
    pub x: Option<i32>,
    pub y: Option<i32>,
    pub pressure: Option<i32>,
    pub tip: Option<bool>,
    pub proximity: Option<bool>,
    pub eraser: Option<bool>,
    pub dropped: bool,
}

pub struct Decoder {
    size: usize,
    bytes: Vec<u8>,
    pending: Report,
    dropped: bool,
}

impl Decoder {
    pub fn new(size: usize) -> Self {
        assert!(size == 16 || size == 24);
        Self {
            size,
            bytes: Vec::new(),
            pending: Report::default(),
            dropped: false,
        }
    }
    pub fn native() -> Self {
        Self::new(if cfg!(target_pointer_width = "32") {
            16
        } else {
            24
        })
    }
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Report> {
        self.bytes.extend_from_slice(bytes);
        let complete = self.bytes.len() / self.size * self.size;
        let mut reports = Vec::new();
        for record in self.bytes[..complete].chunks_exact(self.size) {
            let o = self.size - 8;
            let kind = u16::from_ne_bytes(record[o..o + 2].try_into().unwrap());
            let code = u16::from_ne_bytes(record[o + 2..o + 4].try_into().unwrap());
            let value = i32::from_ne_bytes(record[o + 4..].try_into().unwrap());
            let time = if self.size == 16 {
                i32::from_ne_bytes(record[..4].try_into().unwrap()) as f64 * 1000.0
                    + i32::from_ne_bytes(record[4..8].try_into().unwrap()) as f64 / 1000.0
            } else {
                i64::from_ne_bytes(record[..8].try_into().unwrap()) as f64 * 1000.0
                    + i64::from_ne_bytes(record[8..16].try_into().unwrap()) as f64 / 1000.0
            };
            if kind == 0 && code == 3 {
                self.dropped = true;
                self.pending = Report::default();
                continue;
            }
            if self.dropped {
                if kind == 0 && code == 0 {
                    reports.push(Report {
                        time_ms: time,
                        dropped: true,
                        ..Report::default()
                    });
                    self.dropped = false;
                }
                continue;
            }
            match (kind, code) {
                (3, 0) => self.pending.x = Some(value),
                (3, 1) => self.pending.y = Some(value),
                (3, 0x18) => self.pending.pressure = Some(value),
                (1, 0x14a) => self.pending.tip = Some(value != 0),
                (1, 0x140 | 0x141) => {
                    self.pending.proximity = Some(value != 0);
                    if value != 0 {
                        self.pending.eraser = Some(code == 0x141);
                    }
                }
                (0, 0) => {
                    self.pending.time_ms = time;
                    reports.push(std::mem::take(&mut self.pending));
                }
                _ => {}
            }
        }
        self.bytes.drain(..complete);
        reports
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    None,
    Down,
    Up,
}

#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub edge: Edge,
    pub moved: bool,
    pub x: f64,
    pub y: f64,
    pub time_ms: f64,
    pub eraser: bool,
}

#[derive(Default)]
pub struct Tracker {
    x: i32,
    y: i32,
    pressure: i32,
    tip: bool,
    contact: bool,
    eraser: bool,
}

impl Tracker {
    pub fn frame(&mut self, report: Report, ranges: [i32; 4], screen: (f64, f64)) -> Frame {
        if report.dropped || report.proximity == Some(false) {
            self.tip = false;
            self.pressure = 0;
        }
        if let Some(x) = report.x {
            self.x = x;
        }
        if let Some(y) = report.y {
            self.y = y;
        }
        if let Some(pressure) = report.pressure {
            self.pressure = pressure;
        }
        if let Some(tip) = report.tip {
            self.tip = tip;
        }
        if let Some(eraser) = report.eraser {
            self.eraser = eraser;
        }
        let contact = self.tip || self.pressure >= if self.contact { 20 } else { 40 };
        let edge = match (self.contact, contact) {
            (false, true) => Edge::Down,
            (true, false) => Edge::Up,
            _ => Edge::None,
        };
        self.contact = contact;
        let mapped = |value: i32, minimum: i32, maximum: i32, dimension: f64| {
            (value - minimum) as f64 / (maximum - minimum).max(1) as f64 * dimension
        };
        Frame {
            edge,
            moved: report.x.is_some() || report.y.is_some(),
            x: mapped(self.x, ranges[0], ranges[1], screen.0),
            y: mapped(self.y, ranges[2], ranges[3], screen.1),
            time_ms: report.time_ms,
            eraser: self.eraser,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn event(size: usize, kind: u16, code: u16, value: i32) -> Vec<u8> {
        let mut bytes = vec![0; size - 8];
        bytes.extend(kind.to_ne_bytes());
        bytes.extend(code.to_ne_bytes());
        bytes.extend(value.to_ne_bytes());
        bytes
    }
    #[test]
    fn both_event_abis_preserve_partial_reports_and_pressure_contact() {
        for size in [16, 24] {
            let mut decoder = Decoder::new(size);
            let mut data = event(size, 3, 0, 7812);
            data.extend(event(size, 3, 1, 10416));
            data.extend(event(size, 3, 0x18, 800));
            data.extend(event(size, 0, 0, 0));
            assert!(decoder.feed(&data[..size + 3]).is_empty());
            let reports = decoder.feed(&data[size + 3..]);
            let mut tracker = Tracker::default();
            let frame = tracker.frame(reports[0], [0, 15624, 0, 20832], (930.0, 1240.0));
            assert_eq!(frame.edge, Edge::Down);
            assert_eq!((frame.x, frame.y), (465.0, 620.0));
            let frame = tracker.frame(
                Report {
                    dropped: true,
                    ..Report::default()
                },
                [0, 15624, 0, 20832],
                (930.0, 1240.0),
            );
            assert_eq!(frame.edge, Edge::Up);
        }
    }
    #[test]
    fn dropped_reports_discard_incomplete_position() {
        let mut decoder = Decoder::new(16);
        let mut data = event(16, 3, 0, 99);
        data.extend(event(16, 0, 3, 0));
        data.extend(event(16, 3, 1, 123));
        data.extend(event(16, 0, 0, 0));
        let reports = decoder.feed(&data);
        assert!(reports[0].dropped);
        assert!(reports[0].x.is_none() && reports[0].y.is_none());
    }
}
