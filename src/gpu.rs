//! GPU tile rasterizer (wgpu). wgpu selects Metal on Apple platforms (macOS +
//! iOS), Vulkan on Linux/Android, DX12 on Windows. Strokes the same tile-local
//! polylines as [`crate::render_tile`], but on the GPU and many tiles per render
//! pass, packed into one 8192² `R8Unorm` "meta texture" so the fixed per-dispatch
//! and per-readback overhead amortizes across a whole batch instead of being
//! paid per 512x512 tile. Tiny batches fall back to tiny-skia.
//!
//! Two correctness details make it match tiny-skia:
//!   * Per-cell clip. All tiles share one render pass, so geometry is merely
//!     offset into its cell; the fragment shader discards anything outside its
//!     512x512 cell so a stroke crossing a tile edge can't bleed into the
//!     neighbour (tiny-skia clips each tile to [0, 512)).
//!   * Union coverage, not additive. tiny-skia unions all subpaths and fills
//!     once, so 10 overlapping strokes look identical to 1. A Max blend over
//!     coverage gives the same: each fragment writes `a = opacity*coverage` and
//!     `max()` keeps `opacity * unionCoverage`.
//!
//! Strokes are tessellated to triangles (segment quads + round cap/join discs)
//! with analytic SDF anti-aliasing in the fragment shader.

use crate::raster::{render_cpu, RenderedTile, TileJob};
use crate::TILE_SIZE;
use std::cell::RefCell;
use std::mem::size_of;

const CELL: u32 = TILE_SIZE; // 512
/// The fragment shader's cell clip hardcodes 512.0, so a changed `TILE_SIZE`
/// would silently let strokes bleed between tiles rather than fail to build.
const _: () = assert!(TILE_SIZE == 512);
const MAX_TEX_DIM: u32 = 8192; // safe GPU max; 16 cells per row
const CELLS_PER_ROW: u32 = MAX_TEX_DIM / CELL; // 16
const MAX_TILES_PER_BATCH: usize = (CELLS_PER_ROW * CELLS_PER_ROW) as usize; // 256
/// At or below this many tiles in a chunk, the CPU (tiny-skia) path beats the
/// GPU because the fixed per-pass cost (clear + submit + sync + readback) isn't
/// amortized. See `render_batch`.
const CPU_FALLBACK_MAX_TILES: usize = 2;
/// Vertex-buffer ceiling per render pass. A dense zoom level tessellates far
/// more geometry than 256 tiles' worth fits in one buffer, so batches are cut
/// by byte budget as well as tile count. Capped below the device's
/// `max_buffer_size` to keep the allocation modest on mobile GPUs.
const VERT_BYTE_BUDGET: u64 = 128 << 20; // 128 MB = 4M verts

/// One tile vertex. `pos` is absolute pixel coords in the meta texture; `local`
/// carries the SDF coordinate (segment: signed perp dist in x; disc: offset
/// from center); `cell` is the cell's top-left for per-fragment clipping.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVert {
    pos: [f32; 2],
    local: [f32; 2],
    cell: [f32; 2],
    kind: f32, // 0 = segment, 1 = disc
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    tex_size: [f32; 2],
    half_width: f32,
    aa: f32,
    color: [f32; 4], // straight RGBA, 0..1
}

const WGSL_SRC: &str = r#"
struct Uniforms {
    tex_size: vec2<f32>,
    half_width: f32,
    aa: f32,
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) cell: vec2<f32>,
    @location(2) @interpolate(flat) kind: f32,
};

@vertex
fn vs(
    @location(0) pos: vec2<f32>,
    @location(1) local: vec2<f32>,
    @location(2) cell: vec2<f32>,
    @location(3) kind: f32,
) -> VOut {
    var o: VOut;
    let ndc = vec2<f32>(pos.x / u.tex_size.x * 2.0 - 1.0,
                        1.0 - pos.y / u.tex_size.y * 2.0);
    o.position = vec4<f32>(ndc, 0.0, 1.0);
    o.local = local;
    o.cell = cell;
    o.kind = kind;
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    // Clip to this tile's cell so strokes can't bleed into neighbours.
    let p = in.position.xy;
    if (p.x < in.cell.x || p.x >= in.cell.x + 512.0 ||
        p.y < in.cell.y || p.y >= in.cell.y + 512.0) {
        discard;
    }

    var d: f32; // distance inside the stroke edge (>0 inside)
    if (in.kind > 0.5) {
        d = u.half_width - length(in.local);
    } else {
        d = u.half_width - abs(in.local.x);
    }
    var cov: f32;
    if (u.aa > 0.5) {
        cov = clamp(d + 0.5, 0.0, 1.0);
    } else {
        if (d >= 0.0) { cov = 1.0; } else { cov = 0.0; }
    }
    if (cov <= 0.0) {
        discard;
    }

    // Single-channel coverage into the R8 target; Max blend unions overlapping strokes.
    let a = u.color.a * cov;
    return vec4<f32>(a, 0.0, 0.0, 1.0);
}
"#;

pub struct WgpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    /// Persistent 8192² `R8Unorm` render target (cleared each pass) and its view.
    target_view: wgpu::TextureView,
    target: wgpu::Texture,
    /// Persistent uniform buffer + bind group; uniforms rewritten per pass.
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// CPU tessellation scratch + GPU vertex buffer (grown on demand, never shrunk).
    scratch_verts: RefCell<Vec<GpuVert>>,
    vbuf: RefCell<wgpu::Buffer>,
    /// Mappable staging buffer for texture readback (full 64 MB, allocated once).
    readback: wgpu::Buffer,
    /// Max vertex bytes one pass may upload: `VERT_BYTE_BUDGET` clamped to the
    /// device's `max_buffer_size`.
    max_vert_bytes: u64,
    /// Adapter name/type/backend, surfaced by `TileRasterizer::gpu_info`.
    pub(crate) adapter_info: String,
}

impl WgpuRenderer {
    pub fn new() -> Result<Self, String> {
        // Headless: no surface, so no display handle needed.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("no wgpu adapter: {e}"))?;

        let info = adapter.get_info();
        let adapter_info = format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tile-raster"),
            ..Default::default()
        }))
        .map_err(|e| format!("request_device: {e}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile-shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // GpuVert: pos(8) local(8) cell(8) kind(4) _pad(4) = 32 bytes.
        let vbuf_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 24,
                    shader_location: 3,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tile-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vbuf_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    // Max blend: result = max(src, dst); factors ignored for Max, so
                    // overlapping coverage unions instead of accumulating.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Max,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Max,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("meta-texture"),
            size: wgpu::Extent3d {
                width: MAX_TEX_DIM,
                height: MAX_TEX_DIM,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Small initial vertex buffer; render_chunk grows it to the largest pass.
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verts"),
            size: 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Full-texture staging buffer (64 MB). bytes_per_row = 8192 is 256-aligned.
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (MAX_TEX_DIM * MAX_TEX_DIM) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_vert_bytes = VERT_BYTE_BUDGET.min(device.limits().max_buffer_size);

        Ok(Self {
            device,
            queue,
            pipeline,
            target_view,
            target,
            uniform_buf,
            bind_group,
            scratch_verts: RefCell::new(Vec::new()),
            vbuf: RefCell::new(vbuf),
            readback,
            max_vert_bytes,
            adapter_info,
        })
    }

    /// Rasterize every job with the given stroke width / AA / opacity, batching
    /// up to `MAX_TILES_PER_BATCH` tiles into one meta texture per render pass.
    /// Color RGB is irrelevant (coverage only uses `alpha`); the palette applies
    /// the real color downstream. Returns tiles in input order.
    pub fn render_batch(
        &self,
        jobs: &[TileJob],
        stroke_width: f32,
        anti_alias: bool,
        alpha: u8,
    ) -> Vec<RenderedTile> {
        let half = stroke_width * 0.5;
        let color = [0.0, 0.0, 1.0, alpha as f32 / 255.0];

        let mut out = Vec::with_capacity(jobs.len());
        let mut i = 0;
        while i < jobs.len() {
            let chunk = self.next_chunk(&jobs[i..]);
            i += chunk.len();
            // Tiny chunks don't amortize the fixed per-pass cost (clear + submit +
            // sync + readback), so stroke them on the CPU with tiny-skia instead.
            // A lone job too big for the vertex budget also lands here.
            if chunk.len() <= CPU_FALLBACK_MAX_TILES || vert_bytes(chunk) > self.max_vert_bytes {
                render_cpu(chunk, stroke_width, anti_alias, alpha, &mut out);
            } else {
                self.render_chunk(chunk, half, anti_alias, color, &mut out);
            }
        }
        out
    }

    /// Longest prefix of `rest` that fits one render pass: at most
    /// `MAX_TILES_PER_BATCH` tiles and at most `max_vert_bytes` of geometry.
    /// Always returns at least one job so the caller makes progress.
    fn next_chunk<'j, 'a>(&self, rest: &'j [TileJob<'a>]) -> &'j [TileJob<'a>] {
        let mut bytes = 0u64;
        let mut n = 0;
        while n < rest.len() && n < MAX_TILES_PER_BATCH {
            let jb = vert_bytes(&rest[n..n + 1]);
            if n > 0 && bytes + jb > self.max_vert_bytes {
                break;
            }
            bytes += jb;
            n += 1;
        }
        &rest[..n.max(1)]
    }

    fn render_chunk(
        &self,
        chunk: &[TileJob],
        half: f32,
        aa: bool,
        color: [f32; 4],
        out: &mut Vec<RenderedTile>,
    ) {
        let cols = CELLS_PER_ROW;

        // Geometry: expand by 0.75px when anti-aliasing so the SDF edge has room.
        let pad = if aa { 0.75 } else { 0.0 };
        let mut verts = self.scratch_verts.borrow_mut();
        verts.clear();
        for (i, job) in chunk.iter().enumerate() {
            let cx = ((i as u32 % cols) * CELL) as f32;
            let cy = ((i as u32 / cols) * CELL) as f32;
            tessellate(job.polylines, half + pad, cx, cy, &mut verts);
        }

        // Upload vertices, growing the persistent buffer only when a pass needs more.
        let needed = (verts.len() * size_of::<GpuVert>()) as u64;
        let mut vbuf = self.vbuf.borrow_mut();
        if vbuf.size() < needed.max(1) {
            *vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("verts"),
                size: needed,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !verts.is_empty() {
            self.queue
                .write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        }

        let uniforms = Uniforms {
            tex_size: [MAX_TEX_DIM as f32, MAX_TEX_DIM as f32],
            half_width: half,
            aa: if aa { 1.0 } else { 0.0 },
            color,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Only the rows the batch actually fills need clearing/copying back.
        let rows = chunk.len().div_ceil(cols as usize) as u32;
        let used_h = rows * CELL;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tiles"),
            });
        {
            // Always run the pass so the target is cleared even if verts is empty
            // (avoids reading stale pixels from a prior pass).
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raster"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !verts.is_empty() {
                rpass.set_pipeline(&self.pipeline);
                rpass.set_bind_group(0, &self.bind_group, &[]);
                rpass.set_vertex_buffer(0, vbuf.slice(..));
                rpass.draw(0..verts.len() as u32, 0..1);
            }
        }

        // Copy the used region of the texture into the mappable staging buffer.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(MAX_TEX_DIM), // 8192, 256-aligned
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: MAX_TEX_DIM,
                height: used_h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        // Map the used bytes, slice each cell out (row stride = MAX_TEX_DIM), unmap.
        let used_bytes = (MAX_TEX_DIM as u64) * (used_h as u64);
        let slice = self.readback.slice(0..used_bytes);
        slice.map_async(wgpu::MapMode::Read, |r| {
            r.expect("readback map failed");
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        {
            let data = slice.get_mapped_range();
            let stride = MAX_TEX_DIM as usize;
            for (i, job) in chunk.iter().enumerate() {
                let cx = (i as u32 % cols) as usize * CELL as usize;
                let cy = (i as u32 / cols) as usize * CELL as usize;
                let mut coverage = vec![0u8; (CELL * CELL) as usize];
                for r in 0..CELL as usize {
                    let src = (cy + r) * stride + cx;
                    coverage[r * CELL as usize..(r + 1) * CELL as usize]
                        .copy_from_slice(&data[src..src + CELL as usize]);
                }
                out.push(RenderedTile {
                    x: job.x,
                    y: job.y,
                    coverage,
                });
            }
        }
        self.readback.unmap();
    }
}

/// Upper bound on the vertex bytes `tessellate` emits for these jobs: 6 verts
/// per point (cap/join disc) plus 6 per segment, i.e. at most 12 per point.
/// Only counts lengths, so it's cheap enough to call while forming batches.
fn vert_bytes(jobs: &[TileJob]) -> u64 {
    let points: usize = jobs
        .iter()
        .flat_map(|j| j.polylines.iter())
        .map(|l| l.len())
        .sum();
    (points * 12 * size_of::<GpuVert>()) as u64
}

/// Expand polylines into triangles: each segment becomes a quad; each point a
/// disc for round caps/joins. Coords are offset by (ox, oy) into the meta
/// texture; (ox, oy) is also the cell origin used for per-fragment clipping.
fn tessellate(
    polylines: &[Vec<(f32, f32)>],
    half: f32,
    ox: f32,
    oy: f32,
    verts: &mut Vec<GpuVert>,
) {
    let cell = [ox, oy];
    for line in polylines {
        if line.is_empty() {
            continue;
        }
        // Round caps/joins: a disc at every point.
        for &(px, py) in line {
            push_disc(px + ox, py + oy, half, cell, verts);
        }
        // Segment quads.
        for w in line.windows(2) {
            let (ax, ay) = w[0];
            let (bx, by) = w[1];
            let (dx, dy) = (bx - ax, by - ay);
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-6 {
                continue;
            }
            let (nx, ny) = (-dy / len * half, dx / len * half);
            let a_pos = (ax + ox, ay + oy);
            let b_pos = (bx + ox, by + oy);
            let ap = [a_pos.0 + nx, a_pos.1 + ny];
            let am = [a_pos.0 - nx, a_pos.1 - ny];
            let bp = [b_pos.0 + nx, b_pos.1 + ny];
            let bm = [b_pos.0 - nx, b_pos.1 - ny];
            // local.x = signed perpendicular distance (+half / -half at the edges)
            seg_vert(ap, half, cell, verts);
            seg_vert(am, -half, cell, verts);
            seg_vert(bp, half, cell, verts);
            seg_vert(am, -half, cell, verts);
            seg_vert(bm, -half, cell, verts);
            seg_vert(bp, half, cell, verts);
        }
    }
}

#[inline]
fn seg_vert(pos: [f32; 2], perp: f32, cell: [f32; 2], verts: &mut Vec<GpuVert>) {
    verts.push(GpuVert {
        pos,
        local: [perp, 0.0],
        cell,
        kind: 0.0,
        _pad: 0.0,
    });
}

#[inline]
fn push_disc(cx: f32, cy: f32, r: f32, cell: [f32; 2], verts: &mut Vec<GpuVert>) {
    let corners = [(-r, -r), (r, -r), (r, r), (-r, -r), (r, r), (-r, r)];
    for (lx, ly) in corners {
        verts.push(GpuVert {
            pos: [cx + lx, cy + ly],
            local: [lx, ly],
            cell,
            kind: 1.0,
            _pad: 0.0,
        });
    }
}
