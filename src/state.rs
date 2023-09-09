use std::{future, sync::Arc};

use bytemuck::{Pod, Zeroable};
use glam::Vec4Swizzles;
use rand::Rng;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use wgpu::util::DeviceExt;
use winit::{
    event::{MouseScrollDelta, VirtualKeyCode, WindowEvent},
    window::Window,
};

use crate::{
    camera::{Camera, CameraUniform},
    vertex::{Instance, InstanceRaw, Vertex},
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ParticleCpuData {
    speed: glam::Vec3,
    _unused: f32,
}

struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    cpu_data_buffer: wgpu::Buffer,
}

pub struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instances: Vec<Instance>,
    instances_raw: Vec<InstanceRaw>,
    instances_cpu_data: Vec<ParticleCpuData>,
    instance_buffer: wgpu::Buffer,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    compute_pipeline: Option<ComputePipeline>,
    frame_time_samples: [f32; 25],
    frame_time_index: usize,
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, -0.5, 0.0],
        vertex_position: [0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        vertex_position: [1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.0],
        vertex_position: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.0],
        vertex_position: [1.0, 1.0],
    },
];

const INDICES: &[u16] = &[0, 1, 2, 3, 2, 1];

impl State {
    pub fn new(window: Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, // Vulkan, Metal, DX12, WebGPU
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns both the window and the surface so this is safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                label: Some("4"),
            },
            None,
        ))
        .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|texture_format| texture_format.is_srgb())
            .expect("Did not find an sRGB texture to render to"); // Change here to render HDR

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera = Camera {
            // position the camera one unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 1.0, 5000.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, -100.0).into(),
            // which way is "up"
            up: glam::Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 20.0,
            znear: 0.0,
            zfar: 10000.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::descriptor(), InstanceRaw::descriptor()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None, // 5.
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let index_count = INDICES.len().try_into().unwrap();

        let mut rng = rand::thread_rng();
        let max_particle_count = adapter
            .limits()
            .max_storage_buffer_binding_size
            .min(adapter.limits().max_uniform_buffer_binding_size);

        let instances = (0..1_500_000)
            .map(|_| {
                let x: f32 = (rng.gen::<f32>() - 0.5) * 850.0;
                let y: f32 = (rng.gen::<f32>() - 0.5) * 820.0;
                let z: f32 = (rng.gen::<f32>() - 0.1) * 1000.0;
                let position = glam::Vec3::new(x, y, z);
                let rotation = glam::Quat::from_axis_angle(glam::Vec3::Z, 0.0);
                let color = glam::Vec4::new(
                    0.12 + rng.gen::<f32>() / 4.0 + (x / 850.0 + 0.5) / 2.0,
                    0.75 + rng.gen::<f32>() / 5.0,
                    rng.gen(),
                    1.0,
                );
                Instance {
                    position,
                    rotation,
                    color,
                }
            })
            .collect::<Vec<_>>();

        let instances_cpu_data = (0..instances.len())
            .map(|_| ParticleCpuData {
                speed: glam::Vec3::new(
                    rng.gen::<f32>() - 0.5,
                    rng.gen::<f32>() - 0.5,
                    rng.gen::<f32>() - 0.5,
                )
                .normalize()
                    / 5.0,
                _unused: 0.0,
            })
            .collect::<Vec<_>>();

        let instances_raw = instances
            .par_iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances_raw),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
        });

        let compute_pipeline = Some(Self::create_compute_pipeline(
            &device,
            &instances_cpu_data,
            &instance_buffer,
        ));

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            instances,
            instances_raw,
            instance_buffer,
            instances_cpu_data,
            camera,
            camera_bind_group,
            camera_buffer,
            camera_uniform,
            compute_pipeline,
            frame_time_samples: Default::default(),
            frame_time_index: 0,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn size(&self) -> &winit::dpi::PhysicalSize<u32> {
        &self.size
    }

    pub fn input(&mut self, event: &winit::event::WindowEvent) -> bool {
        if let WindowEvent::MouseWheel { delta, .. } = event {
            let MouseScrollDelta::PixelDelta(pos) = delta else {
                return false;
            };
            self.camera.eye.z += pos.y as f32 / 50.0;
            self.camera.target = self.camera.eye + glam::Vec3::new(0.0, 0.0, -1.0);

            return true;
        }

        if let WindowEvent::KeyboardInput { input, .. } = event {
            if input.virtual_keycode == Some(VirtualKeyCode::R) {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Compute Encoder"),
                        });

                let tmp_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Gang!"),
                    mapped_at_creation: false,
                    size: (std::mem::size_of::<InstanceRaw>() * self.instances_raw.len()) as _,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                }));

                encoder.copy_buffer_to_buffer(
                    &self.instance_buffer,
                    0,
                    &tmp_buffer,
                    0,
                    tmp_buffer.size(),
                );
                self.queue.submit(Some(encoder.finish()));

                let tmp_clone = tmp_buffer.clone();
                // let (sender, receiver) = futures::channel::oneshot::channel::<Vec<InstanceRaw>>();
                tmp_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |x| {
                        x.unwrap();
                        let gpu_data_bytes = tmp_clone
                            .slice(..)
                            .get_mapped_range()
                            .iter()
                            .copied()
                            .collect::<Vec<_>>();
                        let gpu_data: &[InstanceRaw] = bytemuck::cast_slice(&gpu_data_bytes);
                        let gpu_data_vec = gpu_data.to_vec();
                        // println!("YESSIR");
                        // sender.send(gpu_data_vec).unwrap();
                    });

                // println!("Waiting!");
                // self.instances_raw = futures::executor::block_on(receiver).unwrap();
                // println!("Done!");

                for (instance, raw) in self.instances.iter_mut().zip(&self.instances_raw) {
                    instance.position = raw.model.w_axis.xyz();
                }

                self.compute_pipeline = None;
            }
        }
        false
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
        }
    }

    fn move_particles(&mut self) {
        if let Some(compute_pipeline) = &self.compute_pipeline {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });

            {
                let mut raytracing_pass = encoder.begin_compute_pass(&Default::default());
                raytracing_pass.set_pipeline(&compute_pipeline.pipeline);
                raytracing_pass.set_bind_group(0, &compute_pipeline.bind_group, &[]);
                raytracing_pass.dispatch_workgroups(10_000, 150, 1);
            }

            // let tmp = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            //     label: Some("Gang!"),
            //     mapped_at_creation: false,
            //     size: (std::mem::size_of::<InstanceRaw>() * self.instances_raw.len()) as _,
            //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            // }));

            // encoder.copy_buffer_to_buffer(&self.instance_buffer, 0, &tmp, 0, tmp.size());

            self.queue.submit(Some(encoder.finish()));

            // let tmp_clone = tmp.clone();
            // tmp.slice(..).map_async(wgpu::MapMode::Read, move |x| {
            //     x.unwrap();
            //
            //     let a = tmp_clone.slice(..).get_mapped_range().iter().copied().collect::<Vec<_>>();
            //     let b: &[InstanceRaw] = bytemuck::cast_slice(&a);
            //     for c in b {
            //         // println!("gpu transformed: {c}");
            //     }
            // });
            //
            // let a = self.instances
            //     .par_iter_mut()
            //     .zip(&self.instances_cpu_data)
            //     .map(|(instance, cpu_data)| {
            //         instance.position += cpu_data.speed;
            //         instance.to_raw()
            //     })
            //     .collect::<Vec<_>>();
            //
            // for b in a {
            //     // println!("cpu transformed: {b}");
            // }
        } else {
            // Move particles
            self.instances
                .par_iter_mut()
                .zip(&self.instances_cpu_data)
                .map(|(instance, cpu_data)| {
                    instance.position += cpu_data.speed;
                    instance.to_raw()
                })
                .collect_into_vec(&mut self.instances_raw);

            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances_raw),
            );
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let start = std::time::Instant::now();
        self.move_particles();

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoders = vec![];

        let mut render_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let mut render_pass = render_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.index_count, 0, 0..self.instances.len() as _);
        }

        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        encoders.push(render_encoder.finish());
        self.queue.submit(encoders);
        output.present();

        let end = std::time::Instant::now();
        let delta = end - start;
        self.frame_time_samples[self.frame_time_index] = delta.as_micros() as f32;
        self.frame_time_index = (self.frame_time_index + 1) % self.frame_time_samples.len();
        let average_frame_time_us: f32 =
            self.frame_time_samples.iter().sum::<f32>() / self.frame_time_samples.len() as f32;
        println!(
            "Frame time: {}ms | res: {}x{}",
            average_frame_time_us / 1000.0,
            self.size.width,
            self.size.height
        );
        Ok(())
    }

    fn create_compute_pipeline(
        device: &wgpu::Device,
        instances_cpu_data: &[ParticleCpuData],
        instance_buffer: &wgpu::Buffer,
    ) -> ComputePipeline {
        let cpu_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cpu Data Buffer"),
            usage: wgpu::BufferUsages::STORAGE,
            contents: bytemuck::cast_slice(instances_cpu_data),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("1"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            label: Some("2"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cpu_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("3"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute_kernel.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: "main",
        });

        ComputePipeline {
            pipeline,
            bind_group,
            cpu_data_buffer,
        }
    }
}
