//! A persistent daemon connection owns the fast-ink mode and layer hint.
//! Disconnecting releases both, including after a crash or SIGKILL.

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub struct FastInk {
    connection: BufReader<UnixStream>,
    area: Option<[i32; 4]>,
}

impl FastInk {
    pub fn connect(area: [i32; 4]) -> io::Result<Self> {
        let socket = std::env::var_os("EINK_HINTS_SOCKET")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    std::env::var_os("XDG_RUNTIME_DIR").unwrap_or_else(|| "/run/user/0".into()),
                )
                .join("eink-hints.sock")
            });
        Self::connect_at(&socket, area)
    }

    fn connect_at(socket: &Path, area: [i32; 4]) -> io::Result<Self> {
        let connection = UnixStream::connect(socket)?;
        connection.set_read_timeout(Some(Duration::from_secs(3)))?;
        connection.set_write_timeout(Some(Duration::from_secs(3)))?;
        let mut lease = Self {
            connection: BufReader::new(connection),
            area: None,
        };
        lease.set_area(area)?;
        Ok(lease)
    }

    pub fn set_area(&mut self, area: [i32; 4]) -> io::Result<()> {
        if self.area == Some(area) {
            return Ok(());
        }
        let [x, y, width, height] = area;
        let request = serde_json::json!({"layer_area":[x,y,width,height,0], "stylus":true});
        writeln!(self.connection.get_mut(), "{request}")?;
        let mut answer = String::new();
        self.connection.read_line(&mut answer)?;
        if answer.trim() != "ok" {
            return Err(io::Error::other(format!(
                "Fast-ink request failed: {}",
                answer.trim()
            )));
        }
        self.area = Some(area);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    #[test]
    fn hint_and_stylus_lease_lasts_until_disconnect() {
        let path = std::env::temp_dir().join(format!("ink-pad-hints-{}.sock", std::process::id()));
        let listener = UnixListener::bind(&path).unwrap();
        let server = std::thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let mut socket = BufReader::new(socket);
            let mut line = String::new();
            socket.read_line(&mut line).unwrap();
            let request: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(
                request["layer_area"],
                serde_json::json!([0, 1860, 1860, 620, 0])
            );
            assert_eq!(request["stylus"], true);
            socket.get_mut().write_all(b"ok\n").unwrap();
            line.clear();
            assert_eq!(
                socket.read_line(&mut line).unwrap(),
                0,
                "lease releases on EOF"
            );
        });
        let mut lease = FastInk::connect_at(&path, [0, 1860, 1860, 620]).unwrap();
        lease.set_area([0, 1860, 1860, 620]).unwrap(); // unchanged geometry sends nothing
        drop(lease);
        server.join().unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
