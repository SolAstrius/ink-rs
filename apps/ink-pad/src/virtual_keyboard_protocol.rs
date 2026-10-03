//! Protocol from wvkbd v0.20, with its license retained in protocols/.
#![allow(dead_code, non_camel_case_types, unused_unsafe, unused_variables)]
#![allow(non_upper_case_globals, non_snake_case, unused_imports, clippy::all)]

use wayland_client;
use wayland_client::protocol::*;

pub mod __interfaces {
    use wayland_client::protocol::__interfaces::*;
    wayland_scanner::generate_interfaces!("protocols/virtual-keyboard-unstable-v1.xml");
}
use self::__interfaces::*;
wayland_scanner::generate_client_code!("protocols/virtual-keyboard-unstable-v1.xml");
