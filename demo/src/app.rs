use wasm_bindgen::prelude::*;
use web_sys::{Event, HtmlInputElement, HtmlSelectElement, MessageEvent, Worker};
use yew::prelude::*;

use crate::types::{
    BackendName, ExampleName, RunHistoryItem, RunPayload, RunRequest, RunResult, RunStatus,
};

fn gen_job_id() -> String {
    let now = js_sys::Date::now() as u64;
    let rand = (js_sys::Math::random() * 16_777_215.0_f64) as u64;
    format!("{}-{:x}", now, rand)
}

fn default_payload() -> RunPayload {
    RunPayload {
        backend: BackendName::WasmCpu,
        example: ExampleName::Ex1Volume,
        dim: 3,
        nelem: 8,
        p: 2,
        q: 4,
    }
}

/// JS object → JSON string → serde_json::Value
/// 避开 serde_wasm_bindgen 0.6 不支持 deserialize_any 的限制
fn js_to_json(val: JsValue) -> Option<serde_json::Value> {
    let s = js_sys::JSON::stringify(&val).ok()?.as_string()?;
    serde_json::from_str(&s).ok()
}

/// serde type → JSON string → JsValue（供 postMessage 发送）
fn to_js_value<T: serde::Serialize>(v: &T) -> JsValue {
    let s = serde_json::to_string(v).expect("serialize");
    js_sys::JSON::parse(&s).expect("parse JSON")
}

#[function_component(App)]
pub fn app() -> Html {
    // ── 渲染用 state ──────────────────────────────────────────────────────────
    let payload = use_state(default_payload);
    let logs = use_state(Vec::<String>::new);
    let is_running = use_state(|| false);
    let status_text = use_state(|| "Idle".to_string());
    let run_result = use_state(|| Option::<RunResult>::None);
    let run_history = use_state(Vec::<RunHistoryItem>::new);
    let worker: UseStateHandle<Option<Worker>> = use_state(|| None);

    // ── Rc<RefCell<T>> mutable refs ──────────────────────────────────────────
    // use_mut_ref 返回同一个 Rc，长生命周期闭包 borrow() 永远读到最新值，
    // 彻底解决 use_state 闭包陈旧快照（stale closure）问题
    let current_job_id = use_mut_ref(String::new);
    let logs_buf = use_mut_ref(Vec::<String>::new);
    let history_buf = use_mut_ref(Vec::<RunHistoryItem>::new);
    let payload_snap = use_mut_ref(default_payload);

    // ── 初始化 Worker（仅执行一次）───────────────────────────────────────────
    {
        let worker_state = worker.clone();
        let current_job_id = current_job_id.clone();
        let logs_buf = logs_buf.clone();
        let history_buf = history_buf.clone();
        let payload_snap = payload_snap.clone();
        let logs = logs.clone();
        let is_running = is_running.clone();
        let status_text = status_text.clone();
        let run_result = run_result.clone();
        let run_history = run_history.clone();

        use_effect_with((), move |_| {
            let w = Worker::new("/worker/runnerWorker.js")
                .expect("failed to create Worker");

            let cb = Closure::wrap(Box::new(move |event: MessageEvent| {
                // 用 JSON.stringify → serde_json::from_str 解析 JS 对象，
                // 完全绕开 serde_wasm_bindgen 0.6 不支持 deserialize_any 的问题
                let data: serde_json::Value = match js_to_json(event.data()) {
                    Some(v) => v,
                    None => return,
                };

                // 用 jobId 过滤（borrow() 读到 on_run 写入的最新值）
                let msg_job_id = match data.get("jobId").and_then(|v| v.as_str()) {
                    Some(id) => id.to_string(),
                    None => return,
                };
                if msg_job_id != *current_job_id.borrow() {
                    return;
                }

                match data.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "log" => {
                        if let Some(msg) = data.get("message").and_then(|v| v.as_str()) {
                            let msg = msg.to_string();
                            let mut buf = logs_buf.borrow_mut();
                            buf.push(msg.clone());
                            let snapshot = buf.clone();
                            drop(buf);
                            status_text.set(msg);
                            logs.set(snapshot);
                        }
                    }
                    "result" => {
                        is_running.set(false);
                        if let Ok(result) =
                            serde_json::from_value::<RunResult>(data["data"].clone())
                        {
                            {
                                let mut buf = logs_buf.borrow_mut();
                                buf.push(format!(
                                    "Finished in {:.3} ms",
                                    result.duration_ms
                                ));
                                let snapshot = buf.clone();
                                drop(buf);
                                logs.set(snapshot);
                            }
                            {
                                let mut hist = history_buf.borrow_mut();
                                hist.insert(
                                    0,
                                    RunHistoryItem {
                                        id: msg_job_id.clone(),
                                        status: RunStatus::Success,
                                        payload: payload_snap.borrow().clone(),
                                        duration_ms: Some(result.duration_ms),
                                        value: Some(result.value),
                                        error: result.error,
                                        message: None,
                                    },
                                );
                                let snapshot = hist.clone();
                                drop(hist);
                                run_history.set(snapshot);
                            }
                            status_text.set("Done".to_string());
                            run_result.set(Some(result));
                        }
                    }
                    "error" => {
                        is_running.set(false);
                        status_text.set("Failed".to_string());
                        if let Some(msg) = data.get("message").and_then(|v| v.as_str()) {
                            {
                                let mut buf = logs_buf.borrow_mut();
                                buf.push(format!("Error: {}", msg));
                                let snapshot = buf.clone();
                                drop(buf);
                                logs.set(snapshot);
                            }
                            {
                                let mut hist = history_buf.borrow_mut();
                                hist.insert(
                                    0,
                                    RunHistoryItem {
                                        id: msg_job_id.clone(),
                                        status: RunStatus::Failed,
                                        payload: payload_snap.borrow().clone(),
                                        duration_ms: None,
                                        value: None,
                                        error: None,
                                        message: Some(msg.to_string()),
                                    },
                                );
                                let snapshot = hist.clone();
                                drop(hist);
                                run_history.set(snapshot);
                            }
                        }
                        run_result.set(None);
                    }
                    _ => {}
                }
            }) as Box<dyn FnMut(MessageEvent)>);

            w.set_onmessage(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
            worker_state.set(Some(w));
            || {}
        });
    }

    let can_run = !*is_running
        && payload.dim >= 1
        && payload.dim <= 3
        && payload.nelem >= 1
        && payload.p >= 2
        && payload.q >= 1;

    // ── 点击 Run ─────────────────────────────────────────────────────────────
    let on_run = {
        let payload = payload.clone();
        let logs = logs.clone();
        let is_running = is_running.clone();
        let status_text = status_text.clone();
        let run_result = run_result.clone();
        let worker = worker.clone();
        let current_job_id = current_job_id.clone();
        let logs_buf = logs_buf.clone();
        let payload_snap = payload_snap.clone();

        Callback::from(move |_: MouseEvent| {
            if !(!*is_running
                && payload.dim >= 1
                && payload.dim <= 3
                && payload.nelem >= 1
                && payload.p >= 2
                && payload.q >= 1)
            {
                return;
            }
            let w = match &*worker {
                Some(w) => w.clone(),
                None => return,
            };

            let job_id = gen_job_id();
            // 写入 RefCell，消息处理闭包 borrow() 时能读到这个新值
            *current_job_id.borrow_mut() = job_id.clone();

            let initial_logs = vec![
                format!(
                    "Submit {} on {}",
                    payload.example.as_str(),
                    payload.backend.as_str()
                ),
                format!(
                    "dim={}, nelem={}, p={}, q={}",
                    payload.dim, payload.nelem, payload.p, payload.q
                ),
            ];
            *logs_buf.borrow_mut() = initial_logs.clone();
            *payload_snap.borrow_mut() = (*payload).clone();

            is_running.set(true);
            status_text.set("Submitting task".to_string());
            run_result.set(None);
            logs.set(initial_logs);

            let req = RunRequest {
                msg_type: "run".to_string(),
                job_id,
                payload: (*payload).clone(),
            };
            // 用 JSON.parse(JSON.stringify) 发送，确保 worker 收到标准 JS 对象
            w.post_message(&to_js_value(&req)).expect("post_message");
        })
    };

    // ── 清除日志 ──────────────────────────────────────────────────────────────
    let on_clear = {
        let logs = logs.clone();
        let logs_buf = logs_buf.clone();
        Callback::from(move |_: MouseEvent| {
            logs_buf.borrow_mut().clear();
            logs.set(vec![]);
        })
    };

    // ── 参数变更回调 ──────────────────────────────────────────────────────────
    let on_backend_change = Callback::from(|_: Event| {});

    let on_example_change = {
        let payload = payload.clone();
        Callback::from(move |e: Event| {
            let sel: HtmlSelectElement = e.target().unwrap().dyn_into().unwrap();
            let ex = match sel.value().as_str() {
                "ex2_surface" => ExampleName::Ex2Surface,
                "ex3_volume_combined" => ExampleName::Ex3VolumeCombined,
                "poisson" => ExampleName::Poisson,
                _ => ExampleName::Ex1Volume,
            };
            payload.set(RunPayload { example: ex, ..(*payload).clone() });
        })
    };

    let on_dim_change = {
        let payload = payload.clone();
        Callback::from(move |e: Event| {
            let sel: HtmlSelectElement = e.target().unwrap().dyn_into().unwrap();
            let dim = sel.value().parse::<u32>().unwrap_or(3);
            payload.set(RunPayload { dim, ..(*payload).clone() });
        })
    };

    let on_nelem_change = {
        let payload = payload.clone();
        Callback::from(move |e: Event| {
            let inp: HtmlInputElement = e.target().unwrap().dyn_into().unwrap();
            let nelem = inp.value().parse::<u32>().unwrap_or(8);
            payload.set(RunPayload { nelem, ..(*payload).clone() });
        })
    };

    let on_p_change = {
        let payload = payload.clone();
        Callback::from(move |e: Event| {
            let inp: HtmlInputElement = e.target().unwrap().dyn_into().unwrap();
            let p = inp.value().parse::<u32>().unwrap_or(2);
            payload.set(RunPayload { p, ..(*payload).clone() });
        })
    };

    let on_q_change = {
        let payload = payload.clone();
        Callback::from(move |e: Event| {
            let inp: HtmlInputElement = e.target().unwrap().dyn_into().unwrap();
            let q = inp.value().parse::<u32>().unwrap_or(4);
            payload.set(RunPayload { q, ..(*payload).clone() });
        })
    };

    let log_text = if logs.is_empty() {
        "No logs yet.".to_string()
    } else {
        logs.join("\n")
    };

    html! {
        <main class="page">
            <header class="hero">
                <p class="kicker">{"Reed Web Bench"}</p>
                <h1>{"WASM Worker Test Console"}</h1>
                <p class="subtitle">{"选择示例后在 Web Worker 中运行 WASM，查看运行过程、结果与耗时，为后续后端切换性能比较做基线。"}</p>
            </header>

            <section class="panel controls">
                <h2>{"Run Config"}</h2>
                <div class="grid">
                    <label>
                        {"Backend"}
                        <select onchange={on_backend_change}>
                            <option value="wasm-cpu" selected=true>{ payload.backend.label() }</option>
                        </select>
                    </label>

                    <label>
                        {"Example"}
                        <select onchange={on_example_change}>
                            { for ["ex1_volume","ex2_surface","ex3_volume_combined","poisson"].iter().map(|v| {
                                let selected = *v == payload.example.as_str();
                                html! { <option value={*v} selected={selected}>{*v}</option> }
                            }) }
                        </select>
                    </label>

                    <label>
                        {"Dimension"}
                        <select onchange={on_dim_change}>
                            { for [(1u32,"1D"),(2u32,"2D"),(3u32,"3D")].iter().map(|(v, label)| {
                                let selected = *v == payload.dim;
                                html! { <option value={v.to_string()} selected={selected}>{*label}</option> }
                            }) }
                        </select>
                    </label>

                    <label>
                        {"nelem"}
                        <input type="number" min="1" step="1"
                            value={payload.nelem.to_string()}
                            onchange={on_nelem_change} />
                    </label>

                    <label>
                        {"p"}
                        <input type="number" min="2" step="1"
                            value={payload.p.to_string()}
                            onchange={on_p_change} />
                    </label>

                    <label>
                        {"q"}
                        <input type="number" min="1" step="1"
                            value={payload.q.to_string()}
                            onchange={on_q_change} />
                    </label>
                </div>

                <div class="actions">
                    <button class="run" disabled={!can_run} onclick={on_run}>
                        { if *is_running { "Running..." } else { "Run Example" } }
                    </button>
                    <button class="clear" disabled={*is_running} onclick={on_clear}>
                        {"Clear Logs"}
                    </button>
                    <span class="status">{"Status: "}{ &*status_text }</span>
                </div>
            </section>

            <section class="layout">
                <article class="panel logs">
                    <h2>{"Runtime Logs"}</h2>
                    <pre>{ log_text }</pre>
                </article>

                <article class="panel result">
                    <h2>{"Run Result"}</h2>
                    {
                        match &*run_result {
                            Some(r) => html! {
                                <div class="metrics">
                                    <p><strong>{"Example: "}</strong>{ &r.example }</p>
                                    <p><strong>{"Backend: "}</strong>{ &r.backend }</p>
                                    <p><strong>{"Value: "}</strong>{ format!("{:.12}", r.value) }</p>
                                    { r.expected.map(|v| html! {
                                        <p><strong>{"Expected: "}</strong>{ format!("{:.12}", v) }</p>
                                    }).unwrap_or_default() }
                                    { r.error.map(|v| html! {
                                        <p><strong>{"Error: "}</strong>{ format!("{:.6e}", v) }</p>
                                    }).unwrap_or_default() }
                                    <p><strong>{"Duration: "}</strong>{ format!("{:.3} ms", r.duration_ms) }</p>
                                </div>
                            },
                            None => html! { <p class="empty">{"No result yet."}</p> },
                        }
                    }
                </article>
            </section>

            <section class="panel history">
                <h2>{"History (for backend comparison)"}</h2>
                <table>
                    <thead>
                        <tr>
                            <th>{"status"}</th>
                            <th>{"backend"}</th>
                            <th>{"example"}</th>
                            <th>{"dim"}</th>
                            <th>{"nelem/p/q"}</th>
                            <th>{"duration(ms)"}</th>
                            <th>{"error"}</th>
                            <th>{"message"}</th>
                        </tr>
                    </thead>
                    <tbody>
                        { if run_history.is_empty() {
                            html! {
                                <tr><td colspan="8" class="empty">{"No runs yet."}</td></tr>
                            }
                        } else {
                            run_history.iter().map(|item| {
                                let cls = item.status.as_str();
                                html! {
                                    <tr key={item.id.clone()}>
                                        <td class={cls}>{ item.status.as_str() }</td>
                                        <td>{ item.payload.backend.as_str() }</td>
                                        <td>{ item.payload.example.as_str() }</td>
                                        <td>{ item.payload.dim }</td>
                                        <td>{ format!("{}/{}/{}", item.payload.nelem, item.payload.p, item.payload.q) }</td>
                                        <td>{ item.duration_ms.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "-".into()) }</td>
                                        <td>{ item.error.map(|v| format!("{:.3e}", v)).unwrap_or_else(|| "-".into()) }</td>
                                        <td>{ item.message.clone().unwrap_or_else(|| "-".into()) }</td>
                                    </tr>
                                }
                            }).collect::<Html>()
                        }}
                    </tbody>
                </table>
            </section>
        </main>
    }
}
