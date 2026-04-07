use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExampleName {
    Ex1Volume,
    Ex2Surface,
    Ex3VolumeCombined,
    Poisson,
}

impl ExampleName {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExampleName::Ex1Volume => "ex1_volume",
            ExampleName::Ex2Surface => "ex2_surface",
            ExampleName::Ex3VolumeCombined => "ex3_volume_combined",
            ExampleName::Poisson => "poisson",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendName {
    #[serde(rename = "wasm-cpu")]
    WasmCpu,
}

impl BackendName {
    pub fn as_str(&self) -> &'static str {
        "wasm-cpu"
    }

    pub fn label(&self) -> &'static str {
        "WASM CPU (current)"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPayload {
    pub backend: BackendName,
    pub example: ExampleName,
    pub dim: u32,
    pub nelem: u32,
    pub p: u32,
    pub q: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub backend: String,
    pub example: String,
    pub dim: u32,
    pub nelem: u32,
    pub p: u32,
    pub q: u32,
    pub logs: Vec<String>,
    pub value: f64,
    pub expected: Option<f64>,
    pub error: Option<f64>,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(rename = "jobId")]
    pub job_id: String,
    pub payload: RunPayload,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunStatus {
    Success,
    Failed,
}

impl RunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunStatus::Success => "success",
            RunStatus::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunHistoryItem {
    pub id: String,
    pub status: RunStatus,
    pub payload: RunPayload,
    pub duration_ms: Option<f64>,
    pub value: Option<f64>,
    pub error: Option<f64>,
    pub message: Option<String>,
}
