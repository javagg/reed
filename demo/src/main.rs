mod app;
mod types;

use app::App;

fn main() {
    yew::Renderer::<App>::new().render();
}
