mod state;
mod vertex;
mod camera;

use crate::state::State;
use log::warn;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(1500, 900))
        .with_title("Particles!")
        .build(&event_loop)
        .expect("Unable to create Window");

    let mut state = State::new(window);

    event_loop.run(move |event, _, control_fow| match event {
        // Only process the event if the ID is correct
        Event::WindowEvent { event, window_id }
            if state.window().id() == window_id && !state.input(&event) =>
        {
            match event {
                WindowEvent::CloseRequested => {
                    *control_fow = ControlFlow::Exit;
                }
                WindowEvent::Resized(size) => {
                    state.resize(size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(*new_inner_size);
                }
                _ => {}
            }
        },
        Event::RedrawRequested(window_id) if state.window().id() == window_id => {
            if let Err(e) =  state.render() {
                match e {
                    wgpu::SurfaceError::Lost => {
                        warn!("Surface lost, reconfiguring.");
                        state.resize(*state.size());
                    },
                    wgpu::SurfaceError::OutOfMemory => {
                        log::error!("OOM. Exiting.");
                        *control_fow = ControlFlow::Exit;
                    }
                    e => {
                        log::error!("{e}");
                    }
                }
            }
        }
        Event::MainEventsCleared => {
            state.window().request_redraw();
        }
        _ => {}
    });
}
