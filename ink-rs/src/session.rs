//! Reusable scratch buffers and a caller-plus-one-worker inference budget.

use crate::kernels::PreparedInput;
use crate::model::{DirectionScratch, Workspace};
use crate::{error, Backend, Model, Result};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// Repeated inference without reallocating scratch buffers after warmup.
/// One thread uses the caller only. Two threads add one persistent backward
/// worker; every layer shares one immutable packed input between directions.
pub struct Session {
    model: Arc<Model>,
    workspace: Workspace,
    backend: Backend,
    worker: Option<Worker>,
}

impl Session {
    pub fn new(model: Model, backend: Backend, threads: usize) -> Result<Self> {
        if !(1..=2).contains(&threads) {
            return Err(error("Inference supports one or two total compute threads"));
        }
        let backend = backend.resolve()?;
        let model = Arc::new(model);
        let worker = if threads == 2 {
            Some(Worker::new(Arc::clone(&model))?)
        } else {
            None
        };
        Ok(Self {
            model,
            workspace: Workspace::default(),
            backend,
            worker,
        })
    }

    /// Output borrows the session and stays valid until its next mutation.
    /// Recurrent state is reset even though allocation capacity is retained.
    pub fn infer(&mut self, features: &[f32], timesteps: usize) -> Result<&[f32]> {
        self.model.infer_with_workspace(
            features,
            timesteps,
            self.backend,
            &mut self.workspace,
            self.worker.as_ref(),
        )?;
        Ok(&self.workspace.logits)
    }

    pub fn logits(&self) -> &[f32] {
        &self.workspace.logits
    }
    pub fn threads(&self) -> usize {
        if self.worker.is_some() {
            2
        } else {
            1
        }
    }
    pub fn backend(&self) -> Backend {
        self.backend
    }
    pub fn output_width(&self) -> usize {
        self.model.output_width()
    }
    pub fn layer_count(&self) -> usize {
        self.model.layer_count()
    }
}

enum Job {
    Backward {
        layer: usize,
        prepared: Arc<PreparedInput>,
    },
    Stop,
}

pub(crate) struct Worker {
    jobs: mpsc::SyncSender<Job>,
    completed: mpsc::Receiver<()>,
    scratch: Arc<Mutex<DirectionScratch>>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    fn new(model: Arc<Model>) -> Result<Self> {
        let (jobs, receive) = mpsc::sync_channel::<Job>(1);
        let (done, completed) = mpsc::sync_channel::<()>(1);
        let scratch = Arc::new(Mutex::new(DirectionScratch::default()));
        let worker_scratch = Arc::clone(&scratch);
        let handle = std::thread::Builder::new()
            .name("ink-backward".into())
            .spawn(move || {
                while let Ok(job) = receive.recv() {
                    match job {
                        Job::Stop => break,
                        Job::Backward { layer, prepared } => {
                            let Ok(mut scratch) = worker_scratch.lock() else {
                                break;
                            };
                            model.run_backward(layer, &prepared, &mut scratch);
                            drop(scratch);
                            // Release the input before acknowledging. The caller
                            // can then safely repack its uniquely owned buffer.
                            drop(prepared);
                            if done.send(()).is_err() {
                                break;
                            }
                        }
                    }
                }
            })?;
        Ok(Self {
            jobs,
            completed,
            scratch,
            handle: Some(handle),
        })
    }

    pub fn submit(&self, layer: usize, prepared: Arc<PreparedInput>) -> Result<()> {
        self.jobs
            .send(Job::Backward { layer, prepared })
            .map_err(|_| error("Backward worker stopped"))
    }

    pub fn finish(&self) -> Result<MutexGuard<'_, DirectionScratch>> {
        self.completed
            .recv()
            .map_err(|_| error("Backward worker stopped"))?;
        self.scratch
            .lock()
            .map_err(|_| error("Backward worker state is poisoned"))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.jobs.send(Job::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
