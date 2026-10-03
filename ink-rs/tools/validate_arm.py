# /// script
# dependencies = ["numpy", "pyelftools", "unicorn"]
# ///
"""Compile the actual ARMv7 NEON kernels and check them under Unicorn.

This validates generated instructions and ABI handling, not device performance
or full ARMv7 model inference. Requires the installed ARMv7 Rust target and a
host execution environment that permits Unicorn's JIT.
"""
import argparse
from pathlib import Path
import json
import struct
import subprocess
import tempfile
import numpy as np
from elftools.elf.elffile import ELFFile
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_ARM
from unicorn.arm_const import UC_ARM_REG_C1_C0_2, UC_ARM_REG_FPEXC, UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_SP
from unicorn import arm_const


def compile_kernels(crate, out):
    # Expose the private module only in this temporary test translation unit.
    source = (crate / 'src/kernels.rs').read_text().replace('mod neon {', 'pub(crate) mod neon {')
    kernels = out / 'kernels.rs'
    kernels.write_text(source)
    wrapper = out / 'wrapper.rs'
    wrapper.write_text('''
#[derive(Debug)] pub struct Error(String);
pub type Result<T> = std::result::Result<T, Error>;
fn error(message: impl Into<String>) -> Error { Error(message.into()) }
mod kernels { include!("kernels.rs"); }
#[no_mangle]
pub unsafe extern "C" fn ink_test_tile(w: *const f32, x: *const f32, y: *mut f32, k: usize, stride: usize) {
    unsafe { kernels::neon::tile4x8(w, x, y, k, stride); }
}
#[no_mangle]
pub unsafe extern "C" fn ink_test_affine(p: *const f32, b: *const f32, u: *const f32, h: *const f32, y: *mut f32) {
    unsafe { kernels::neon::affine4(p, b, u, h, y); }
}
#[no_mangle]
pub unsafe extern "C" fn ink_test_cell(f: *const f32, c: *const f32, i: *const f32, g: *const f32, y: *mut f32) {
    unsafe { kernels::neon::cell4(f, c, i, g, y); }
}
#[no_mangle]
pub unsafe extern "C" fn ink_test_mul(a: *const f32, b: *const f32, y: *mut f32) {
    unsafe { kernels::neon::mul4(a, b, y); }
}
#[no_mangle]
pub unsafe extern "C" fn ink_test_sigmoid(values: *mut f32) {
    unsafe { kernels::neon::activation4(values, false); }
}
#[no_mangle]
pub unsafe extern "C" fn ink_test_tanh(values: *mut f32) {
    unsafe { kernels::neon::activation4(values, true); }
}
''')
    obj = out / 'kernels.o'
    subprocess.run(['rustc', '--edition=2021', '--crate-type=lib', '--target', 'armv7-unknown-linux-musleabihf',
                    '--emit=obj', '-C', 'opt-level=3', '-C', 'panic=abort', '-A', 'warnings', str(wrapper), '-o', str(obj)], check=True)
    return obj


class EmulatedKernels:
    def __init__(self, obj):
        self.uc = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        self.uc.mem_map(0x10000, 0x100000)
        self.uc.mem_map(0x300000, 0x10000)
        self.uc.mem_map(0x400000, 0x400000)
        self.stop = 0x10000
        self.uc.mem_write(self.stop, b'\x1e\xff\x2f\xe1')
        self.uc.reg_write(UC_ARM_REG_C1_C0_2, 0xF << 20)
        self.uc.reg_write(UC_ARM_REG_FPEXC, 1 << 30)
        with obj.open('rb') as file:
            elf = ELFFile(file)
            assert elf['e_machine'] == 'EM_ARM'
            offsets = {}
            cursor = 0x11000
            for i, section in enumerate(elf.iter_sections()):
                if section['sh_flags'] & 2 and not section.name.startswith(('.ARM.exidx', '.ARM.extab')):
                    cursor = (cursor + 15) & ~15
                    offsets[i] = cursor
                    self.uc.mem_write(cursor, section.data())
                    cursor += section['sh_size']
            symtab = elf.get_section_by_name('.symtab')
            addresses = {}
            self.functions = {}
            for i, sym in enumerate(symtab.iter_symbols()):
                section_index = sym['st_shndx']
                if isinstance(section_index, int) and section_index in offsets:
                    addresses[i] = offsets[section_index] + sym['st_value']
                    if sym.name.startswith('ink_test_'):
                        self.functions[sym.name] = addresses[i]
            for section in elf.iter_sections():
                if section['sh_type'] != 'SHT_REL' or section['sh_info'] not in offsets:
                    continue
                for relocation in section.iter_relocations():
                    place = offsets[section['sh_info']] + relocation['r_offset']
                    target = addresses.get(relocation['r_info_sym'])
                    if target is None:
                        raise RuntimeError(f'Kernel has an external dependency: {symtab.get_symbol(relocation["r_info_sym"]).name}')
                    kind = relocation['r_info_type']
                    instruction, = struct.unpack('<I', self.uc.mem_read(place, 4))
                    if kind in (28, 29):  # R_ARM_CALL, R_ARM_JUMP24
                        addend = (instruction & 0xFFFFFF) << 2
                        if addend & (1 << 25):
                            addend -= 1 << 26
                        delta = target + addend - place
                        assert delta % 4 == 0
                        instruction = (instruction & 0xFF000000) | ((delta >> 2) & 0xFFFFFF)
                    elif kind == 2:  # R_ARM_ABS32
                        instruction += target
                    elif kind == 3:  # R_ARM_REL32, including constant tables
                        instruction = (instruction + target - place) & 0xFFFFFFFF
                    else:
                        raise RuntimeError(f'Unsupported relocation {kind}')
                    self.uc.mem_write(place, struct.pack('<I', instruction))
        assert len(self.functions) == 6
        self.cursor = 0x400000
        self.abi_checked = set()

    def array(self, values):
        values = np.asarray(values, dtype='<f4')
        self.cursor = (self.cursor + 15) & ~15
        pointer = self.cursor
        self.uc.mem_write(pointer, values.tobytes())
        self.cursor += values.nbytes
        return pointer

    def call(self, name, args):
        sp = 0x30FF00
        # Numerical outputs alone missed an ABI bug: Q-only clobbers left
        # D8-D15 unsaved, corrupting caller state between recognition requests.
        saved = [(getattr(arm_const, f'UC_ARM_REG_R{i}'), 0x72640000 + i) for i in range(4, 12)]
        saved += [(getattr(arm_const, f'UC_ARM_REG_D{i}'), int.from_bytes(struct.pack('<d', 10.0 + i), 'little')) for i in range(8, 16)]
        for register, value in saved:
            self.uc.reg_write(register, value)
        for i, arg in enumerate(args[:4]):
            self.uc.reg_write(UC_ARM_REG_R0 + i, int(arg))
        for i, arg in enumerate(args[4:]):
            self.uc.mem_write(sp + 4 * i, struct.pack('<I', int(arg)))
        self.uc.reg_write(UC_ARM_REG_SP, sp)
        self.uc.reg_write(UC_ARM_REG_LR, self.stop)
        self.uc.emu_start(self.functions[name], self.stop, count=1_000_000)
        assert self.uc.reg_read(UC_ARM_REG_PC) == self.stop
        assert self.uc.reg_read(UC_ARM_REG_SP) == sp, (name, 'stack pointer changed')
        for register, value in saved:
            assert self.uc.reg_read(register) == value, (name, 'callee-saved register changed', register)
        self.abi_checked.add(name)

    def read(self, pointer, shape):
        return np.frombuffer(self.uc.mem_read(pointer, int(np.prod(shape)) * 4), dtype='<f4').reshape(shape).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    crate = Path(__file__).resolve().parents[1]
    out = args.output or Path(tempfile.mkdtemp(prefix='ink-arm-kernels-'))
    out.mkdir(parents=True, exist_ok=True)
    emu = EmulatedKernels(compile_kernels(crate, out))
    rng = np.random.default_rng(407)
    report = {'scope': 'Generated ARMv7 NEON numerical outputs and callee-saved registers under emulation; no device timing.', 'tiles': [], 'state_helpers': {}}
    for k in (1, 10, 37, 432, 560):
        for stride in (8, 16, 248, 320):
            for column in sorted({0, stride - 8}):
                emu.cursor = 0x400000
                x = rng.normal(0, 0.25, (4, k)).astype(np.float32)
                w = rng.normal(0, 0.25, (k, 8)).astype(np.float32)
                initial = np.full((4, stride), 123.0, np.float32)
                xp, wp, yp = emu.array(x), emu.array(w), emu.array(initial)
                emu.call('ink_test_tile', [wp, xp, yp + column * 4, k, stride])
                actual = emu.read(yp, initial.shape)
                expected = initial.copy()
                expected[:, column:column + 8] = 0
                for i in range(k):
                    expected[:, column:column + 8] += x[:, i, None] * w[None, i, :]
                max_error = float(abs(actual - expected).max())
                assert max_error <= 5e-5, (k, stride, column, max_error)
                report['tiles'].append({'inputs': k, 'stride': stride, 'column': column, 'max_error': max_error})
    p, b, u, h = rng.normal(size=(4, 4)).astype(np.float32)
    for name, values, expected in [
        ('affine', [p, b, u, h], (p + b) + u * h),
        ('cell', [p, b, u, h], p * b + u * h),
        ('mul', [p, b], p * b),
    ]:
        emu.cursor = 0x400000
        pointers = [emu.array(value) for value in values]
        output = emu.array(np.zeros(4, np.float32))
        emu.call('ink_test_' + name, pointers + [output])
        actual = emu.read(output, (4,))
        max_error = float(abs(actual - expected).max())
        assert max_error <= 1e-6, (name, actual, expected)
        report['state_helpers'][name] = {'max_error': max_error}
    inputs = np.r_[np.linspace(-32, 32, 8193), [-128, -87, -0.0, 0.0, 1e-12, -1e-12, 87, 128, np.inf, -np.inf]].astype(np.float32)
    report['activations'] = {}
    for name in ('sigmoid', 'tanh'):
        maximum = 0.0
        for start in range(0, len(inputs), 4):
            chunk = inputs[start:start + 4]
            emu.cursor = 0x400000
            values = np.zeros(4, np.float32)
            values[:len(chunk)] = chunk
            pointer = emu.array(values)
            emu.call('ink_test_' + name, [pointer])
            actual = emu.read(pointer, (4,))[:len(chunk)]
            expected = (np.tanh(chunk.astype(np.float64)) if name == 'tanh' else 1.0 / (1.0 + np.exp(-chunk.astype(np.float64)))).astype(np.float32)
            maximum = max(maximum, float(abs(actual - expected).max()))
            assert np.isfinite(actual).all()
            if name == 'tanh':
                assert np.array_equal(actual[chunk == 0].view(np.uint32), chunk[chunk == 0].view(np.uint32))
        assert maximum <= 3e-7, (name, maximum)
        report['activations'][name] = {'cases': len(inputs), 'max_error': maximum}
    report['abi_preservation'] = {'functions': sorted(emu.abi_checked), 'registers': 'R4-R11, D8-D15 and SP', 'passed': True}
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(f'ARMv7 generated NEON: {len(report["tiles"])} matrix tiles, 3 recurrent helpers and 2 vector activations passed; activation errors {report["activations"]}')
    print('Results:', out / 'results.json')


if __name__ == '__main__':
    main()
