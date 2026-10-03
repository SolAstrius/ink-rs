//! Bounds-checked access to the subset of FlatBuffers used by these TFLite packs.

use crate::{error, Result};

#[derive(Clone, Copy)]
pub(crate) struct Reader<'a> {
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy)]
pub(crate) struct Table<'a> {
    reader: Reader<'a>,
    pos: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct Vector<'a> {
    reader: Reader<'a>,
    pos: usize,
    pub len: usize,
    width: usize,
}

impl<'a> Reader<'a> {
    pub fn bytes(self, pos: usize, len: usize) -> Result<&'a [u8]> {
        let end = pos
            .checked_add(len)
            .ok_or_else(|| error("FlatBuffer offset overflow"))?;
        self.bytes
            .get(pos..end)
            .ok_or_else(|| error("Truncated FlatBuffer"))
    }

    pub fn u16(self, pos: usize) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(pos, 2)?.try_into().unwrap()))
    }

    pub fn u32(self, pos: usize) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(pos, 4)?.try_into().unwrap()))
    }

    pub fn i32(self, pos: usize) -> Result<i32> {
        Ok(i32::from_le_bytes(self.bytes(pos, 4)?.try_into().unwrap()))
    }

    fn indirect(self, pos: usize) -> Result<usize> {
        let dest = pos
            .checked_add(self.u32(pos)? as usize)
            .ok_or_else(|| error("FlatBuffer offset overflow"))?;
        self.bytes(dest, 4)?;
        Ok(dest)
    }

    pub fn root(self) -> Result<Table<'a>> {
        if self.bytes(4, 4)? != b"TFL3" {
            return Err(error("Expected a TFL3 model"));
        }
        Ok(Table {
            reader: self,
            pos: self.indirect(0)?,
        })
    }
}

impl<'a> Table<'a> {
    fn field(self, index: usize) -> Result<Option<usize>> {
        let distance = self.reader.i32(self.pos)? as i64;
        let vpos = (self.pos as i64)
            .checked_sub(distance)
            .ok_or_else(|| error("Invalid vtable"))?;
        let vpos = usize::try_from(vpos).map_err(|_| error("Invalid vtable"))?;
        let vlen = self.reader.u16(vpos)? as usize;
        let offset = index
            .checked_mul(2)
            .and_then(|v| v.checked_add(4))
            .ok_or_else(|| error("Field overflow"))?;
        self.reader.bytes(vpos, vlen)?;
        if offset + 2 > vlen {
            return Ok(None);
        }
        let relative = self.reader.u16(vpos + offset)? as usize;
        if relative == 0 {
            return Ok(None);
        }
        let object_len = self.reader.u16(vpos + 2)? as usize;
        if relative >= object_len {
            return Err(error("Field outside FlatBuffer table"));
        }
        let pos = self
            .pos
            .checked_add(relative)
            .ok_or_else(|| error("Field overflow"))?;
        self.reader.bytes(self.pos, object_len)?;
        Ok(Some(pos))
    }

    pub fn u32(self, index: usize, default: u32) -> Result<u32> {
        self.field(index)?
            .map_or(Ok(default), |pos| self.reader.u32(pos))
    }

    pub fn i32(self, index: usize, default: i32) -> Result<i32> {
        self.field(index)?
            .map_or(Ok(default), |pos| self.reader.i32(pos))
    }

    pub fn i8(self, index: usize, default: i8) -> Result<i8> {
        self.field(index)?
            .map_or(Ok(default), |pos| Ok(self.reader.bytes(pos, 1)?[0] as i8))
    }

    pub fn table(self, index: usize) -> Result<Option<Table<'a>>> {
        self.field(index)?
            .map(|pos| {
                Ok(Table {
                    reader: self.reader,
                    pos: self.reader.indirect(pos)?,
                })
            })
            .transpose()
    }

    pub fn vector(self, index: usize, width: usize) -> Result<Option<Vector<'a>>> {
        let Some(pos) = self.field(index)? else {
            return Ok(None);
        };
        let start = self.reader.indirect(pos)?;
        let len = self.reader.u32(start)? as usize;
        let size = len
            .checked_mul(width)
            .ok_or_else(|| error("Vector length overflow"))?;
        self.reader.bytes(start + 4, size)?;
        Ok(Some(Vector {
            reader: self.reader,
            pos: start + 4,
            len,
            width,
        }))
    }

    pub fn required_vector(self, index: usize, width: usize) -> Result<Vector<'a>> {
        self.vector(index, width)?
            .ok_or_else(|| error(format!("Missing vector field {index}")))
    }

    pub fn string(self, index: usize) -> Result<Option<&'a str>> {
        self.vector(index, 1)?
            .map(|v| std::str::from_utf8(v.bytes()?).map_err(|_| error("Invalid UTF-8 string")))
            .transpose()
    }
}

impl<'a> Vector<'a> {
    fn element(self, index: usize) -> Result<usize> {
        if index >= self.len {
            return Err(error("FlatBuffer vector index out of range"));
        }
        Ok(self.pos + index * self.width)
    }

    pub fn table(self, index: usize) -> Result<Table<'a>> {
        Ok(Table {
            reader: self.reader,
            pos: self.reader.indirect(self.element(index)?)?,
        })
    }

    pub fn i32(self, index: usize) -> Result<i32> {
        self.reader.i32(self.element(index)?)
    }

    pub fn f32(self, index: usize) -> Result<f32> {
        Ok(f32::from_bits(self.reader.u32(self.element(index)?)?))
    }

    pub fn i64(self, index: usize) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.reader
                .bytes(self.element(index)?, 8)?
                .try_into()
                .unwrap(),
        ))
    }

    pub fn bytes(self) -> Result<&'a [u8]> {
        self.reader.bytes(self.pos, self.len * self.width)
    }
}
