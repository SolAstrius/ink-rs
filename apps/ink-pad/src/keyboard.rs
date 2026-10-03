// SPDX-License-Identifier: GPL-3.0-only
// Adapted from wvkbd v0.20 keyboard handling (Copyright 2020 John Sullivan).
// This Rust implementation batches Unicode text into temporary XKB keymaps.
//! Encode recognized Unicode text as virtual-keyboard events.
use std::io;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditingKey {
    Backspace,
    Enter,
    DeleteWord,
    Left,
    Right,
    Up,
    Down,
}

pub enum InputOperation<'a> {
    Text(&'a str),
    Key(EditingKey),
}

// Printable evdev positions, avoiding Enter, modifiers and function keys.
const PRINTABLE: [u32; 48] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57,
];

pub struct KeyBatch {
    pub keymap: String,
    pub codes: Vec<u32>,
}

fn keymap(symbols: &[(u32, String)]) -> String {
    let mut map =
        String::from("xkb_keymap {\nxkb_keycodes \"ink-pad\" { minimum = 8; maximum = 255;\n");
    for (i, (code, _)) in symbols.iter().enumerate() {
        map.push_str(&format!("<I{i:03X}> = {};\n", code + 8));
    }
    map.push_str("};\nxkb_types \"ink-pad\" { type \"ONE_LEVEL\" { modifiers = None; map[None] = Level1; }; };\nxkb_compatibility \"ink-pad\" {};\nxkb_symbols \"ink-pad\" {\n");
    for (i, (_, symbol)) in symbols.iter().enumerate() {
        map.push_str(&format!(
            "key <I{i:03X}> {{ type[Group1] = \"ONE_LEVEL\", symbols[Group1] = [ {symbol} ] }};\n"
        ));
    }
    if let Some(i) = symbols.iter().position(|(_, symbol)| symbol == "Control_L") {
        map.push_str(&format!("modifier_map Control {{ <I{i:03X}> }};\n"));
    }
    map.push_str("};\n};\n\0");
    map
}

pub fn text_batches(text: &str) -> io::Result<Vec<KeyBatch>> {
    if text.chars().any(char::is_control) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Recognized text must be one line",
        ));
    }
    let mut batches = Vec::new();
    let mut characters = Vec::new();
    let mut codes = Vec::new();
    let finish = |characters: &[char], codes: Vec<u32>| KeyBatch {
        keymap: keymap(
            &characters
                .iter()
                .enumerate()
                .map(|(i, ch)| (PRINTABLE[i], format!("U{:04X}", *ch as u32)))
                .collect::<Vec<_>>(),
        ),
        codes,
    };
    for ch in text.chars() {
        let slot = if let Some(slot) = characters.iter().position(|old| *old == ch) {
            slot
        } else {
            if characters.len() == PRINTABLE.len() {
                batches.push(finish(&characters, std::mem::take(&mut codes)));
                characters.clear();
            }
            characters.push(ch);
            characters.len() - 1
        };
        codes.push(PRINTABLE[slot]);
    }
    if !codes.is_empty() {
        batches.push(finish(&characters, codes));
    }
    Ok(batches)
}

#[cfg(target_os = "linux")]
mod wayland {
    use super::*;
    use crate::virtual_keyboard_protocol::{
        zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
        zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
    };
    use std::{
        fs::File,
        io::Write,
        os::fd::{AsFd, FromRawFd},
        time::Instant,
    };
    use wayland_client::{protocol::wl_seat::WlSeat, Dispatch, QueueHandle};

    pub struct TextKeyboard {
        proxy: ZwpVirtualKeyboardV1,
        current_map: Option<String>,
        clock: Instant,
    }

    impl TextKeyboard {
        pub fn new<D: Dispatch<ZwpVirtualKeyboardV1, ()> + 'static>(
            manager: &ZwpVirtualKeyboardManagerV1,
            seat: &WlSeat,
            qh: &QueueHandle<D>,
        ) -> Self {
            Self {
                proxy: manager.create_virtual_keyboard(seat, qh, ()),
                current_map: None,
                clock: Instant::now(),
            }
        }

        fn upload(&mut self, map: String) -> io::Result<()> {
            if self.current_map.as_ref() != Some(&map) {
                // SAFETY: a successfully created memfd becomes an owned File.
                let raw =
                    unsafe { libc::memfd_create(c"ink-pad-keymap".as_ptr(), libc::MFD_CLOEXEC) };
                if raw < 0 {
                    return Err(io::Error::last_os_error());
                }
                let mut file = unsafe { File::from_raw_fd(raw) };
                file.write_all(map.as_bytes())?;
                self.proxy.keymap(1, file.as_fd(), map.len() as u32);
                self.current_map = Some(map);
            }
            Ok(())
        }

        fn tap(&self, code: u32) {
            let time = self.clock.elapsed().as_millis() as u32;
            self.proxy.key(time, code, 1);
            self.proxy.key(time, code, 0);
        }

        fn send(&mut self, batch: KeyBatch) -> io::Result<()> {
            self.upload(batch.keymap)?;
            self.proxy.modifiers(0, 0, 0, 0);
            for code in batch.codes {
                self.tap(code);
            }
            Ok(())
        }

        pub fn type_text(&mut self, text: &str) -> io::Result<()> {
            for batch in text_batches(text)? {
                self.send(batch)?;
            }
            Ok(())
        }

        pub fn backspace(&mut self) -> io::Result<()> {
            self.editing_key(EditingKey::Backspace)
        }

        pub fn editing_key(&mut self, key: EditingKey) -> io::Result<()> {
            if key == EditingKey::DeleteWord {
                self.upload(keymap(&[
                    (29, "Control_L".into()),
                    (14, "BackSpace".into()),
                ]))?;
                let time = self.clock.elapsed().as_millis() as u32;
                self.proxy.modifiers(0, 0, 0, 0);
                self.proxy.key(time, 29, 1);
                self.proxy.modifiers(1 << 2, 0, 0, 0); // real XKB Control modifier
                self.tap(14);
                self.proxy.key(time, 29, 0);
                self.proxy.modifiers(0, 0, 0, 0);
                return Ok(());
            }
            let (code, symbol) = match key {
                EditingKey::Backspace => (14, "BackSpace"),
                EditingKey::Enter => (28, "Return"),
                EditingKey::Left => (105, "Left"),
                EditingKey::Right => (106, "Right"),
                EditingKey::Up => (103, "Up"),
                EditingKey::Down => (108, "Down"),
                EditingKey::DeleteWord => unreachable!(),
            };
            self.send(KeyBatch {
                keymap: keymap(&[(code, symbol.into())]),
                codes: vec![code],
            })
        }
    }

    impl Drop for TextKeyboard {
        fn drop(&mut self) {
            self.proxy.destroy();
        }
    }

    #[derive(Default)]
    struct SenderState {
        manager: Option<ZwpVirtualKeyboardManagerV1>,
        seat: Option<WlSeat>,
    }
    impl Dispatch<wayland_client::protocol::wl_registry::WlRegistry, ()> for SenderState {
        fn event(
            state: &mut Self,
            registry: &wayland_client::protocol::wl_registry::WlRegistry,
            event: wayland_client::protocol::wl_registry::Event,
            _: &(),
            _: &wayland_client::Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wayland_client::protocol::wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "zwp_virtual_keyboard_manager_v1" => {
                        state.manager = Some(registry.bind(name, 1, qh, ()))
                    }
                    "wl_seat" if state.seat.is_none() => {
                        state.seat = Some(registry.bind(name, version.min(7), qh, ()))
                    }
                    _ => {}
                }
            }
        }
    }
    wayland_client::delegate_noop!(SenderState: ignore ZwpVirtualKeyboardManagerV1);
    wayland_client::delegate_noop!(SenderState: ignore ZwpVirtualKeyboardV1);
    wayland_client::delegate_noop!(SenderState: ignore WlSeat);

    pub fn type_once(text: &str) -> Result<(), Box<dyn std::error::Error>> {
        send_once(&[InputOperation::Text(text)])
    }

    pub fn send_once(operations: &[InputOperation<'_>]) -> Result<(), Box<dyn std::error::Error>> {
        let connection = wayland_client::Connection::connect_to_env()?;
        let mut queue = connection.new_event_queue::<SenderState>();
        let qh = queue.handle();
        let mut state = SenderState::default();
        connection.display().get_registry(&qh, ());
        queue.roundtrip(&mut state)?;
        let mut keyboard = TextKeyboard::new(
            state
                .manager
                .as_ref()
                .ok_or("Wayland virtual keyboard unavailable")?,
            state.seat.as_ref().ok_or("Wayland seat unavailable")?,
            &qh,
        );
        for operation in operations {
            match operation {
                InputOperation::Text(text) => keyboard.type_text(text)?,
                InputOperation::Key(key) => keyboard.editing_key(*key)?,
            }
        }
        queue.roundtrip(&mut state)?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
pub use wayland::{send_once, type_once, TextKeyboard};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_unicode_keeps_character_order_and_case() {
        let text = "Aba привет Ёё 🙂!";
        let batches = text_batches(text).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].codes.len(), text.chars().count());
        assert!(batches[0].keymap.contains("U043F"));
        assert!(batches[0].keymap.contains("U1F642"));
        assert!(batches[0].keymap.ends_with('\0'));
        let repeated = text_batches("aba").unwrap();
        assert_eq!(repeated[0].codes, vec![2, 3, 2]);
    }
    #[test]
    fn batches_large_alphabets_and_rejects_control_keys_before_sending() {
        let text: String = (0x400..0x470).filter_map(char::from_u32).collect();
        let batches = text_batches(&text).unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(
            batches.iter().map(|b| b.codes.len()).sum::<usize>(),
            text.chars().count()
        );
        assert!(text_batches("text\ncommand").is_err());
        assert!(text_batches("\0").is_err());
        assert!(text_batches("").unwrap().is_empty());
    }
}
