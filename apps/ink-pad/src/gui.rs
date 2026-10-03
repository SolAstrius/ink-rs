//! Hand-painted Wayland SHM surface and passive evdev pen input, following
//! scribble-rs. The surface is a bottom-quarter layer rather than fullscreen.

use ink_inference::features::Stroke;
use ink_pad::virtual_keyboard_protocol::{
    zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
    zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
};
use ink_pad::{
    draw::Renderer,
    eink::FastInk,
    engine::{Language, Recognizer},
    input::{Decoder, Tracker},
    keyboard::EditingKey,
    keyboard::TextKeyboard,
    pad::{Action, Pad},
};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use wayland_client::protocol::{
    wl_buffer::{self, WlBuffer},
    wl_callback::{self, WlCallback},
    wl_compositor::WlCompositor,
    wl_output::{self, WlOutput},
    wl_registry::{self, WlRegistry},
    wl_seat::WlSeat,
    wl_shm::{Format, WlShm},
    wl_shm_pool::WlShmPool,
    wl_surface::WlSurface,
};
use wayland_client::{delegate_noop, Connection, Dispatch, QueueHandle, WEnum};
use wayland_protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::{Layer, ZwlrLayerShellV1},
    zwlr_layer_surface_v1::{self, Anchor, KeyboardInteractivity, ZwlrLayerSurfaceV1},
};

#[repr(C)]
#[derive(Default)]
struct AbsInfo {
    value: i32,
    minimum: i32,
    maximum: i32,
    fuzz: i32,
    flat: i32,
    resolution: i32,
}

struct Digitizer {
    file: File,
    ranges: [i32; 4],
}
impl Digitizer {
    fn open(path: Option<PathBuf>) -> io::Result<Self> {
        let path = if let Some(path) = path {
            path
        } else {
            let mut events: Vec<_> = std::fs::read_dir("/sys/class/input")?
                .filter_map(Result::ok)
                .collect();
            events.sort_by_key(|entry| entry.file_name());
            events
                .into_iter()
                .find_map(|entry| {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if !name.starts_with("event") {
                        return None;
                    }
                    let device = std::fs::read_to_string(entry.path().join("device/name")).ok()?;
                    (device.contains("Wacom") && device.contains("Digitizer"))
                        .then(|| PathBuf::from("/dev/input").join(name.as_ref()))
                })
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        "Wacom digitizer not found; use --device",
                    )
                })?
        };
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(&path)?;
        let mut axes = [AbsInfo::default(), AbsInfo::default()];
        for (index, axis) in axes.iter_mut().enumerate() {
            let request = (2u32 << 30)
                | ((std::mem::size_of::<AbsInfo>() as u32) << 16)
                | (u32::from(b'E') << 8)
                | (0x40 + index as u32);
            // SAFETY: EVIOCGABS writes one correctly sized repr(C) input_absinfo.
            if unsafe { libc::ioctl(file.as_raw_fd(), request as _, axis as *mut AbsInfo) } < 0 {
                return Err(io::Error::last_os_error());
            }
            if axis.maximum <= axis.minimum {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid digitizer axis range",
                ));
            }
        }
        eprintln!(
            "ink-pad: passive digitizer {}, raw x={}..{}, y={}..{}",
            path.display(),
            axes[0].minimum,
            axes[0].maximum,
            axes[1].minimum,
            axes[1].maximum
        );
        Ok(Self {
            file,
            ranges: [
                axes[0].minimum,
                axes[0].maximum,
                axes[1].minimum,
                axes[1].maximum,
            ],
        })
    }
}

struct Buffer {
    wl: WlBuffer,
    offset: usize,
    busy: bool,
    stale: Option<[f64; 4]>,
}
struct Buffers {
    memory: *mut u8,
    length: usize,
    width: u32,
    height: u32,
    generation: u32,
    buffers: [Buffer; 2],
}
impl Buffers {
    fn new(
        shm: &WlShm,
        qh: &QueueHandle<App>,
        width: u32,
        height: u32,
        generation: u32,
    ) -> io::Result<Self> {
        let size = width as usize * height as usize * 4;
        // SAFETY: Linux fd syscalls; OwnedFd closes the successfully created fd.
        let fd = unsafe {
            let raw = libc::memfd_create(c"ink-pad".as_ptr(), libc::MFD_CLOEXEC);
            if raw < 0 {
                return Err(io::Error::last_os_error());
            }
            OwnedFd::from_raw_fd(raw)
        };
        if unsafe { libc::ftruncate(fd.as_raw_fd(), (size * 2) as libc::off_t) } < 0 {
            return Err(io::Error::last_os_error());
        }
        let memory = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size * 2,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if memory == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let pool = shm.create_pool(fd.as_fd(), (size * 2) as i32, qh, ());
        let buffers = std::array::from_fn(|index| Buffer {
            wl: pool.create_buffer(
                (index * size) as i32,
                width as i32,
                height as i32,
                width as i32 * 4,
                Format::Argb8888,
                qh,
                (generation, index),
            ),
            offset: index * size,
            busy: false,
            stale: None,
        });
        pool.destroy();
        Ok(Self {
            memory: memory.cast(),
            length: size * 2,
            width,
            height,
            generation,
            buffers,
        })
    }
}
impl Drop for Buffers {
    fn drop(&mut self) {
        for buffer in &self.buffers {
            buffer.wl.destroy();
        }
        unsafe {
            libc::munmap(self.memory.cast(), self.length);
        }
    }
}

struct Request {
    generation: u64,
    language: Language,
    strokes: Vec<Stroke>,
    automatic: bool,
}
struct Response {
    generation: u64,
    result: Result<String, String>,
    elapsed: Duration,
    automatic: bool,
}
struct Worker {
    sender: Option<mpsc::SyncSender<Request>>,
    receiver: mpsc::Receiver<Response>,
    handle: Option<std::thread::JoinHandle<()>>,
}
impl Worker {
    fn new(packs: PathBuf, beam: f64) -> io::Result<Self> {
        let (sender, requests) = mpsc::sync_channel::<Request>(1);
        let (results, receiver) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("ink-recognizer".into())
            .spawn(move || {
                let mut loaded = None::<(Language, Recognizer)>;
                while let Ok(request) = requests.recv() {
                    let start = Instant::now();
                    let result = (|| {
                        if loaded
                            .as_ref()
                            .is_none_or(|(language, _)| *language != request.language)
                        {
                            loaded = None; // release the previous language's model/LM memory
                            loaded = Some((
                                request.language,
                                Recognizer::load(&packs, request.language, beam)
                                    .map_err(|e| e.to_string())?,
                            ));
                        }
                        let recognition = loaded
                            .as_mut()
                            .unwrap()
                            .1
                            .recognize(&request.strokes)
                            .map_err(|e| e.to_string())?;
                        recognition
                            .hypotheses
                            .first()
                            .map(|h| h.text.clone())
                            .ok_or_else(|| "No recognition hypotheses".to_string())
                    })();
                    if results
                        .send(Response {
                            generation: request.generation,
                            result,
                            elapsed: start.elapsed(),
                            automatic: request.automatic,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            receiver,
            handle: Some(handle),
        })
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct App {
    keyboard_manager: Option<ZwpVirtualKeyboardManagerV1>,
    seat: Option<WlSeat>,
    keyboard: Option<TextKeyboard>,
    compositor: Option<WlCompositor>,
    shm: Option<WlShm>,
    shell: Option<ZwlrLayerShellV1>,
    output: Option<WlOutput>,
    surface: Option<WlSurface>,
    layer: Option<ZwlrLayerSurfaceV1>,
    physical: (u32, u32),
    scale: u32,
    configured: bool,
    running: bool,
    damage: Option<[f64; 4]>,
    frame_pending: bool,
    frame_started: Instant,
    generation: u32,
    buffers: Option<Buffers>,
    renderer: Renderer,
    canvas: Option<tiny_skia::Pixmap>,
    rebuild_canvas: bool,
    status_dirty: bool,
    fast_ink: Option<FastInk>,
    pad: Pad,
    digitizer: Digitizer,
    decoder: Decoder,
    tracker: Tracker,
    worker: Worker,
    pending_generation: Option<u64>,
    pending_after: Vec<Action>,
    last_submitted: u64,
}

impl App {
    fn damage_region(&mut self, bounds: [f64; 4]) {
        self.damage = Some(if let Some(old) = self.damage {
            [
                old[0].min(bounds[0]),
                old[1].min(bounds[1]),
                old[2].max(bounds[2]),
                old[3].max(bounds[3]),
            ]
        } else {
            bounds
        });
        if let Some(buffers) = &mut self.buffers {
            for buffer in &mut buffers.buffers {
                buffer.stale = Some(if let Some(old) = buffer.stale {
                    [
                        old[0].min(bounds[0]),
                        old[1].min(bounds[1]),
                        old[2].max(bounds[2]),
                        old[3].max(bounds[3]),
                    ]
                } else {
                    bounds
                });
            }
        }
    }
    fn damage_all(&mut self) {
        self.rebuild_canvas = true;
        self.damage_region([0.0, 0.0, self.pad.width, self.pad.height]);
    }
    fn damage_status(&mut self) {
        self.status_dirty = true;
        self.damage_region([0.0, self.pad.height - 28.0, self.pad.width, self.pad.height]);
    }
    fn geometry(&mut self) {
        let width = self.physical.0 as f64 / self.scale as f64;
        let height = self.physical.1 as f64 / self.scale as f64;
        if self.pad.width != width || self.pad.screen_height != height {
            self.pad.clear();
            self.pad.width = width;
            self.pad.screen_height = height;
            self.pad.height = height / 4.0;
            self.buffers = None;
            self.damage_all();
            if let Some(layer) = &self.layer {
                layer.set_size(0, self.pad.height.round() as u32);
                layer.set_exclusive_zone(self.pad.height.round() as i32);
            }
        }
    }
    fn action(&mut self, action: Action) {
        match action {
            Action::Recognize => self.recognize(false),
            Action::Clear => {
                self.pad.clear();
                self.pending_after.clear();
                self.pending_generation = None;
            }
            Action::Close => self.running = false,
            Action::Enter if self.pad.busy || !self.pad.strokes.is_empty() => {
                self.pending_after.push(Action::Enter);
                self.pending_generation = Some(self.pad.generation);
                if !self.pad.busy {
                    self.recognize(false);
                }
            }
            Action::Enter => self.send_key(EditingKey::Enter),
            Action::Space => {
                if let Some(keyboard) = &mut self.keyboard {
                    if let Err(error) = keyboard.type_text(" ") {
                        eprintln!("ink-pad: input: {error}");
                        self.pad.result = format!("Input failed: {error}");
                        self.damage_status();
                    }
                }
            }
            Action::DeleteWord => self.send_key(EditingKey::DeleteWord),
            Action::Left => self.send_key(EditingKey::Left),
            Action::Right => self.send_key(EditingKey::Right),
            Action::Up => self.send_key(EditingKey::Up),
            Action::Down => self.send_key(EditingKey::Down),
            Action::ToggleAuto => {
                self.pad.toggle_auto(Instant::now());
                self.pad.result = if self.pad.auto_delay_ms == 0 {
                    "Auto insertion off".into()
                } else {
                    format!("Auto insertion after {} ms", self.pad.auto_delay_ms)
                };
                self.damage_status();
            }
            Action::Backspace => self.send_key(EditingKey::Backspace),
            Action::English | Action::Russian => {
                self.pad.clear();
                self.pending_after.clear();
                self.pending_generation = None;
                self.pad.language = if action == Action::English {
                    Language::English
                } else {
                    Language::Russian
                };
            }
        }
        if matches!(action, Action::Clear | Action::English | Action::Russian) {
            self.damage_all();
        }
    }
    fn send_key(&mut self, key: EditingKey) {
        if let Some(keyboard) = &mut self.keyboard {
            if let Err(error) = keyboard.editing_key(key) {
                eprintln!("ink-pad: input: {error}");
                self.pad.result = format!("Input failed: {error}");
                self.damage_status();
            }
        }
    }
    fn recognize(&mut self, automatic: bool) {
        if self.pad.busy || self.pad.is_down() || self.pad.strokes.is_empty() {
            return;
        }
        let request = Request {
            generation: self.pad.generation,
            language: self.pad.language,
            strokes: self.pad.strokes.clone(),
            automatic,
        };
        if self
            .worker
            .sender
            .as_ref()
            .unwrap()
            .try_send(request)
            .is_ok()
        {
            self.pad.busy = true;
            self.last_submitted = self.pad.generation;
            self.damage_status();
        }
    }
    fn read_pen(&mut self) -> io::Result<()> {
        let mut bytes = [0u8; 24 * 128];
        loop {
            let size = match self.digitizer.file.read(&mut bytes) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Digitizer disconnected",
                    ))
                }
                Ok(size) => size,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(e),
            };
            for report in self.decoder.feed(&bytes[..size]) {
                let frame = self.tracker.frame(
                    report,
                    self.digitizer.ranges,
                    (self.pad.width, self.pad.screen_height),
                );
                let before = self.pad.generation;
                let previous = self.pad.current.last().copied();
                let had_status = !self.pad.result.is_empty();
                let action = self.pad.handle(frame);
                if let Some(action) = action {
                    self.action(action);
                }
                if before != self.pad.generation {
                    if action.is_none() {
                        if frame.eraser {
                            self.damage_all();
                        } else if let Some(point) = self.pad.current.last().copied().or_else(|| {
                            self.pad
                                .strokes
                                .last()
                                .and_then(|stroke| stroke.last())
                                .copied()
                        }) {
                            let previous = previous.unwrap_or(point);
                            if !self.rebuild_canvas {
                                if let Some(canvas) = &mut self.canvas {
                                    Renderer::segment(canvas, previous, point, self.scale);
                                }
                            }
                            self.damage_region([
                                previous.x.min(point.x) - 3.0,
                                previous.y.min(point.y) - 3.0,
                                previous.x.max(point.x) + 3.0,
                                previous.y.max(point.y) + 3.0,
                            ]);
                            if had_status {
                                self.damage_status();
                            }
                        }
                    }
                    if !self.pending_after.is_empty() {
                        self.pending_generation = Some(self.pad.generation);
                    }
                }
                if report.dropped {
                    eprintln!("ink-pad: dropped input events; incomplete stroke ended");
                }
            }
        }
        Ok(())
    }
    fn tick(&mut self) {
        while let Ok(response) = self.worker.receiver.try_recv() {
            self.pad.busy = false;
            if response.generation != self.pad.generation {
                if self.pending_generation == Some(self.pad.generation) && !self.pad.is_down() {
                    self.recognize(false);
                }
                self.damage_status();
                continue;
            }
            let following = if self.pending_generation == Some(response.generation) {
                self.pending_generation = None;
                std::mem::take(&mut self.pending_after)
            } else {
                Vec::new()
            };
            let enter_after = following.contains(&Action::Enter);
            match response.result {
                Ok(text) => {
                    println!("{text}");
                    let _ = io::stdout().flush();
                    if text.is_empty() {
                        self.pad.result = "No text recognized".into();
                    } else if let Some(keyboard) = &mut self.keyboard {
                        let mut output = text.clone();
                        if response.automatic
                            && !enter_after
                            && !text.ends_with(char::is_whitespace)
                        {
                            output.push(' ');
                        }
                        match keyboard.type_text(&output) {
                            Ok(()) => {
                                self.pad.inserted(text);
                                self.damage_all();
                            }
                            Err(error) => {
                                eprintln!("ink-pad: input: {error}");
                                self.pad.result = format!("Input failed: {error}");
                            }
                        }
                    } else {
                        self.pad.result = text;
                    }
                }
                Err(error) => {
                    eprintln!("ink-pad: {error}");
                    self.pad.result = error;
                }
            }
            for action in following {
                match action {
                    Action::Space => {
                        if let Some(keyboard) = &mut self.keyboard {
                            if let Err(error) = keyboard.type_text(" ") {
                                eprintln!("ink-pad: input: {error}");
                                self.pad.result = format!("Input failed: {error}");
                            }
                        }
                    }
                    Action::Enter => self.send_key(EditingKey::Enter),
                    _ => {}
                }
            }
            eprintln!(
                "ink-pad: recognition {:.0} ms",
                response.elapsed.as_secs_f64() * 1000.0
            );
            self.damage_status();
        }
        if self.pad.auto_ready(Instant::now(), self.last_submitted) {
            self.recognize(true);
        }
        if self.frame_pending && self.frame_started.elapsed() > Duration::from_millis(700) {
            self.frame_pending = false;
        }
    }
    fn draw(&mut self, qh: &QueueHandle<Self>) -> io::Result<()> {
        if !self.configured || self.damage.is_none() || self.frame_pending {
            return Ok(());
        }
        let width = (self.pad.width * self.scale as f64).round() as u32;
        let height = (self.pad.height * self.scale as f64).round() as u32;
        let area = [
            0,
            (self.physical.1 - height) as i32,
            width as i32,
            height as i32,
        ];
        if let Some(fast_ink) = &mut self.fast_ink {
            fast_ink.set_area(area)?;
        } else {
            self.fast_ink = Some(FastInk::connect(area)?);
            eprintln!("ink-pad: Y1 layer hint and STYLUS lease active");
        }
        if self
            .buffers
            .as_ref()
            .is_none_or(|b| b.width != width || b.height != height)
        {
            self.generation += 1;
            self.buffers = Some(Buffers::new(
                self.shm.as_ref().unwrap(),
                qh,
                width,
                height,
                self.generation,
            )?);
            self.damage_all();
        }
        if self.rebuild_canvas || self.canvas.is_none() {
            self.canvas = Some(self.renderer.render(&self.pad, self.scale));
            self.rebuild_canvas = false;
            self.status_dirty = false;
        } else if self.status_dirty {
            if let Some(canvas) = &mut self.canvas {
                self.renderer.status(canvas, &self.pad, self.scale);
            }
            self.status_dirty = false;
        }
        let buffers = self.buffers.as_mut().unwrap();
        let Some(index) = buffers.buffers.iter().position(|b| !b.busy) else {
            return Ok(());
        };
        let pixels = self.canvas.as_ref().unwrap();
        let target = unsafe {
            std::slice::from_raw_parts_mut(
                buffers.memory.add(buffers.buffers[index].offset),
                width as usize * height as usize * 4,
            )
        };
        let stale = buffers.buffers[index].stale.take().unwrap();
        let scale = self.scale as f64;
        let left = (stale[0] * scale).floor().clamp(0.0, width as f64) as usize;
        let top = (stale[1] * scale).floor().clamp(0.0, height as f64) as usize;
        let right = (stale[2] * scale).ceil().clamp(0.0, width as f64) as usize;
        let bottom = (stale[3] * scale).ceil().clamp(0.0, height as f64) as usize;
        for row in top..bottom {
            let start = (row * width as usize + left) * 4;
            let end = (row * width as usize + right) * 4;
            target[start..end].copy_from_slice(&pixels.data()[start..end]);
        }
        let surface = self.surface.as_ref().unwrap();
        surface.set_buffer_scale(self.scale as i32);
        surface.attach(Some(&buffers.buffers[index].wl), 0, 0);
        let bounds = self.damage.unwrap();
        let scale = self.scale as f64;
        let left = (bounds[0] * scale).floor().clamp(0.0, width as f64) as i32;
        let top = (bounds[1] * scale).floor().clamp(0.0, height as f64) as i32;
        let right = (bounds[2] * scale).ceil().clamp(0.0, width as f64) as i32;
        let bottom = (bounds[3] * scale).ceil().clamp(0.0, height as f64) as i32;
        surface.damage_buffer(left, top, right - left, bottom - top);
        surface.frame(qh, ());
        surface.commit();
        buffers.buffers[index].busy = true;
        self.damage = None;
        self.frame_pending = true;
        self.frame_started = Instant::now();
        Ok(())
    }
}

impl Dispatch<WlRegistry, ()> for App {
    fn event(
        app: &mut Self,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    app.compositor = Some(registry.bind(name, version.min(4), qh, ()))
                }
                "wl_shm" => app.shm = Some(registry.bind(name, 1, qh, ())),
                "zwp_virtual_keyboard_manager_v1" => {
                    app.keyboard_manager = Some(registry.bind(name, 1, qh, ()))
                }
                "wl_seat" if app.seat.is_none() => {
                    app.seat = Some(registry.bind(name, version.min(7), qh, ()))
                }
                "zwlr_layer_shell_v1" => {
                    app.shell = Some(registry.bind(name, version.min(4), qh, ()))
                }
                "wl_output" if app.output.is_none() => {
                    app.output = Some(registry.bind(name, version.min(2), qh, ()))
                }
                _ => {}
            }
        }
    }
}
impl Dispatch<WlOutput, ()> for App {
    fn event(
        app: &mut Self,
        _: &WlOutput,
        event: wl_output::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_output::Event::Scale { factor } => app.scale = factor.max(1) as u32,
            wl_output::Event::Mode {
                flags: WEnum::Value(flags),
                width,
                height,
                ..
            } if flags.contains(wl_output::Mode::Current) => {
                app.physical = (width.max(1) as u32, height.max(1) as u32)
            }
            _ => {}
        }
        app.geometry();
    }
}
impl Dispatch<ZwlrLayerSurfaceV1, ()> for App {
    fn event(
        app: &mut Self,
        layer: &ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_layer_surface_v1::Event::Configure {
                serial,
                width,
                height,
            } => {
                layer.ack_configure(serial);
                if width > 0 {
                    app.pad.width = width as f64;
                }
                if height > 0 {
                    app.pad.height = height as f64;
                }
                app.configured = true;
                app.damage_all();
            }
            zwlr_layer_surface_v1::Event::Closed => app.running = false,
            _ => {}
        }
    }
}
impl Dispatch<WlBuffer, (u32, usize)> for App {
    fn event(
        app: &mut Self,
        _: &WlBuffer,
        event: wl_buffer::Event,
        data: &(u32, usize),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_buffer::Event::Release = event {
            if let Some(buffers) = &mut app.buffers {
                if buffers.generation == data.0 {
                    buffers.buffers[data.1].busy = false;
                }
            }
        }
    }
}
impl Dispatch<WlCallback, ()> for App {
    fn event(
        app: &mut Self,
        _: &WlCallback,
        event: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            app.frame_pending = false;
        }
    }
}
delegate_noop!(App: ignore WlCompositor);
delegate_noop!(App: ignore WlShm);
delegate_noop!(App: ignore WlShmPool);
delegate_noop!(App: ignore WlSurface);
delegate_noop!(App: ignore ZwlrLayerShellV1);
delegate_noop!(App: ignore WlSeat);
delegate_noop!(App: ignore ZwpVirtualKeyboardManagerV1);
delegate_noop!(App: ignore ZwpVirtualKeyboardV1);

pub fn run(
    packs: PathBuf,
    language: Language,
    beam: f64,
    device: Option<PathBuf>,
    auto_delay_ms: u64,
    typing: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Both input panels anchor at the bottom. Preserve wvkbd's process/keymap
    // while hiding its surface so the raw pen mapping stays on this panel.
    if typing && std::path::Path::new("/usr/local/bin/osk").is_file() {
        let _ = std::process::Command::new("/usr/local/bin/osk")
            .arg("hide")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    let digitizer = Digitizer::open(device)?;
    let connection = Connection::connect_to_env()?;
    let mut queue = connection.new_event_queue::<App>();
    let qh = queue.handle();
    // Block signals before spawning the recognition worker; both threads inherit
    // the mask and the UI handles SIGINT/SIGTERM via its signalfd.
    let signal = unsafe {
        let mut mask: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut mask);
        libc::sigaddset(&mut mask, libc::SIGINT);
        libc::sigaddset(&mut mask, libc::SIGTERM);
        if libc::sigprocmask(libc::SIG_BLOCK, &mask, std::ptr::null_mut()) < 0 {
            return Err(io::Error::last_os_error().into());
        }
        let fd = libc::signalfd(-1, &mask, libc::SFD_CLOEXEC | libc::SFD_NONBLOCK);
        if fd < 0 {
            return Err(io::Error::last_os_error().into());
        }
        OwnedFd::from_raw_fd(fd)
    };
    let mut app = App {
        keyboard_manager: None,
        seat: None,
        keyboard: None,
        compositor: None,
        shm: None,
        shell: None,
        output: None,
        surface: None,
        layer: None,
        physical: (1860, 2480),
        scale: 2,
        configured: false,
        running: true,
        damage: Some([0.0, 0.0, 930.0, 310.0]),
        frame_pending: false,
        frame_started: Instant::now(),
        generation: 0,
        buffers: None,
        renderer: Renderer::default(),
        canvas: None,
        rebuild_canvas: true,
        status_dirty: false,
        fast_ink: None,
        pad: Pad::new(930.0, 1240.0, language),
        digitizer,
        decoder: Decoder::native(),
        tracker: Tracker::default(),
        worker: Worker::new(packs, beam)?,
        pending_generation: None,
        pending_after: Vec::new(),
        last_submitted: 0,
    };
    app.pad.auto_delay_ms = if typing { auto_delay_ms } else { 0 };
    connection.display().get_registry(&qh, ());
    queue.roundtrip(&mut app)?;
    queue.roundtrip(&mut app)?;
    app.pad.typing = typing;
    if typing {
        app.keyboard = Some(TextKeyboard::new(
            app.keyboard_manager
                .as_ref()
                .ok_or("Wayland virtual keyboard unavailable")?,
            app.seat.as_ref().ok_or("Wayland seat unavailable")?,
            &qh,
        ));
    } else {
        app.pad.result = "Write, then tap Recognize.".into();
    }
    let compositor = app
        .compositor
        .as_ref()
        .ok_or("Wayland compositor unavailable")?;
    let shell = app
        .shell
        .as_ref()
        .ok_or("Wayland layer-shell unavailable")?;
    let surface = compositor.create_surface(&qh, ());
    let layer = shell.get_layer_surface(
        &surface,
        app.output.as_ref(),
        Layer::Overlay,
        "ink-pad".into(),
        &qh,
        (),
    );
    layer.set_anchor(Anchor::Bottom | Anchor::Left | Anchor::Right);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer.set_size(0, app.pad.height.round() as u32);
    layer.set_exclusive_zone(app.pad.height.round() as i32);
    app.surface = Some(surface);
    app.layer = Some(layer);
    app.surface.as_ref().unwrap().commit();
    while app.running {
        queue.dispatch_pending(&mut app)?;
        app.tick();
        app.draw(&qh)?;
        match queue.flush() {
            Ok(()) => {}
            Err(wayland_client::backend::WaylandError::Io(e))
                if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e.into()),
        }
        let Some(guard) = queue.prepare_read() else {
            continue;
        };
        let mut fds = [
            libc::pollfd {
                fd: guard.connection_fd().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: app.digitizer.file.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: signal.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        if unsafe { libc::poll(fds.as_mut_ptr(), 3, 16) } < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(io::Error::last_os_error().into());
        }
        if fds[0].revents & libc::POLLIN != 0 {
            match guard.read() {
                Ok(_) => {}
                Err(wayland_client::backend::WaylandError::Io(e))
                    if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e.into()),
            }
        } else {
            drop(guard);
        }
        if fds[0].revents & (libc::POLLERR | libc::POLLHUP) != 0 {
            break;
        }
        if fds[1].revents & (libc::POLLERR | libc::POLLHUP) != 0 {
            return Err("Digitizer disconnected".into());
        }
        if fds[1].revents & libc::POLLIN != 0 {
            app.read_pen()?;
        }
        if fds[2].revents & libc::POLLIN != 0 {
            app.running = false;
        }
    }
    Ok(())
}
