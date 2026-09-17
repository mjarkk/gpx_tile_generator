//! Batched tile rasterization + single-channel PNG encode.
//!
//! [`TileRasterizer`] wraps the wgpu GPU renderer (when available) and falls
//! back to the CPU path otherwise (wasm, `gpu` feature off, GPU init failure on
//! a device without Vulkan/Metal/DX12). Both produce [`TILE_SIZE`]² single-byte
//! coverage, which [`encode_indexed_coverage`] turns into a paletted PNG whose
//! one byte/pixel IS the coverage.
//!
//! Because the color lives entirely in that palette, [`recolor_indexed_png`]
//! repaints an already-encoded tile by rewriting 768 bytes — no re-render and no
//! re-compression.

use crate::{TileStyle, TILE_SIZE};

/// Default track color, and the palette every [`encode_indexed_coverage`]
/// tile carries.
const DEFAULT_RGB: (u8, u8, u8) = (0x00, 0x7A, 0xFF);

/// The 8 bytes every PNG starts with.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// One tile to rasterize: its `(x, y)` key and its tile-local polylines.
pub struct TileJob<'a> {
    /// Tile column at the target zoom.
    pub x: u32,
    /// Tile row at the target zoom.
    pub y: u32,
    /// Polylines in tile-local pixel coords, as [`crate::bucketize_track`] emits.
    pub polylines: &'a [Vec<(f32, f32)>],
}

/// A rendered tile: the same key, plus [`TILE_SIZE`]² single-byte coverage
/// values (unioned stroke alpha, `0..=255`).
///
/// The color is applied later by the indexed-PNG palette, so this feeds
/// [`encode_indexed_coverage`] directly.
pub struct RenderedTile {
    /// Tile column, copied from the [`TileJob`].
    pub x: u32,
    /// Tile row, copied from the [`TileJob`].
    pub y: u32,
    /// `TILE_SIZE * TILE_SIZE` coverage bytes, row-major.
    pub coverage: Vec<u8>,
}

/// Batched tile rasterizer. Prefers the GPU (wgpu); transparently uses the CPU
/// path when the GPU is unavailable.
///
/// Construction is expensive (device + 64 MB of persistent buffers), rendering
/// is not — build one and reuse it for every batch.
pub struct TileRasterizer {
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    gpu: Option<crate::gpu::WgpuRenderer>,
    gpu_error: Option<String>,
}

impl TileRasterizer {
    /// Build a rasterizer, initializing the GPU if possible.
    ///
    /// Never fails: a missing or broken GPU just means every batch routes
    /// through the CPU path. [`TileRasterizer::is_gpu`] reports which path you
    /// got and [`TileRasterizer::gpu_error`] why, for callers that want to log
    /// it — this crate never writes to stderr itself.
    pub fn new() -> Self {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            match crate::gpu::WgpuRenderer::new() {
                Ok(r) => Self {
                    gpu: Some(r),
                    gpu_error: None,
                },
                Err(e) => Self {
                    gpu: None,
                    gpu_error: Some(e),
                },
            }
        }
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        {
            Self {
                gpu_error: Some("built without GPU support".to_string()),
            }
        }
    }

    /// Why the GPU path was unavailable, or `None` when it is in use.
    pub fn gpu_error(&self) -> Option<&str> {
        self.gpu_error.as_deref()
    }

    /// True when batches run on the GPU.
    pub fn is_gpu(&self) -> bool {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            self.gpu.is_some()
        }
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        {
            false
        }
    }

    /// Name of the GPU adapter in use, e.g. `"Apple M1 Pro (IntegratedGpu, Metal)"`.
    ///
    /// `None` on the CPU path.
    pub fn gpu_info(&self) -> Option<&str> {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            self.gpu.as_ref().map(|g| g.adapter_info.as_str())
        }
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        {
            None
        }
    }

    /// Rasterize every job with the given stroke width / AA / opacity, returning
    /// tiles in input order.
    ///
    /// Color is not a parameter: coverage is color-free, and
    /// [`encode_indexed_coverage_rgb`] applies the real color downstream. Only
    /// `alpha` (opacity, `0..=255`) is baked into the coverage.
    pub fn render_batch(
        &self,
        jobs: &[TileJob],
        stroke_width: f32,
        anti_alias: bool,
        alpha: u8,
    ) -> Vec<RenderedTile> {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        if let Some(gpu) = &self.gpu {
            return gpu.render_batch(jobs, stroke_width, anti_alias, alpha);
        }
        let mut out = Vec::with_capacity(jobs.len());
        render_cpu(jobs, stroke_width, anti_alias, alpha, &mut out);
        out
    }
}

impl Default for TileRasterizer {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU rasterization producing the same coverage the GPU path does: tiny-skia
/// gives premultiplied RGBA whose alpha channel equals `opacity * coverage`, so
/// the alpha channel alone is the coverage we want.
pub(crate) fn render_cpu(
    jobs: &[TileJob],
    stroke_width: f32,
    anti_alias: bool,
    alpha: u8,
    out: &mut Vec<RenderedTile>,
) {
    // RGB is irrelevant here (we keep only the alpha channel); the real color is
    // applied by the indexed palette. Opacity (`alpha`) IS what matters.
    let (r, g, b) = DEFAULT_RGB;
    let style = TileStyle {
        color_rgba: (r, g, b, alpha),
        stroke_width,
        anti_alias,
    };
    for job in jobs {
        let pixmap = crate::render_tile(job.polylines, &style);
        let rgba = pixmap.take();
        let coverage: Vec<u8> = (0..(TILE_SIZE * TILE_SIZE) as usize)
            .map(|i| rgba[i * 4 + 3])
            .collect();
        out.push(RenderedTile {
            x: job.x,
            y: job.y,
            coverage,
        });
    }
}

/// Encode a [`TILE_SIZE`]² single-channel coverage tile as an **indexed** PNG,
/// ~5x faster to encode than RGBA and ~74% the size, decoding to a visually
/// identical image.
///
/// Paints the tracks in the default blue; see [`encode_indexed_coverage_rgb`]
/// to pick the color. Panics if `coverage.len() != TILE_SIZE * TILE_SIZE`.
pub fn encode_indexed_coverage(coverage: &[u8]) -> Vec<u8> {
    encode_indexed_coverage_rgb(coverage, DEFAULT_RGB)
}

/// [`encode_indexed_coverage`] in an arbitrary color.
///
/// Every track pixel is `rgb` with `alpha = coverage`, so the one coverage byte
/// per pixel carries all information. We emit a paletted PNG whose pixel byte IS
/// the coverage, with a 256-entry all-`rgb` palette and an identity `tRNS` alpha
/// table. The decoder reconstructs `(r, g, b, a)`.
///
/// The color therefore lives in the palette only, which is what lets
/// [`recolor_indexed_png`] repaint a finished tile without touching its pixels.
///
/// Panics if `coverage.len() != TILE_SIZE * TILE_SIZE`.
pub fn encode_indexed_coverage_rgb(coverage: &[u8], rgb: (u8, u8, u8)) -> Vec<u8> {
    // Palette: 256 identical entries; tRNS: identity, so index a -> (r, g, b, a).
    let (r, g, b) = rgb;
    let mut palette = [0u8; 256 * 3];
    let mut trns = [0u8; 256];
    for a in 0..256 {
        palette[a * 3] = r;
        palette[a * 3 + 1] = g;
        palette[a * 3 + 2] = b;
        trns[a] = a as u8;
    }
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, TILE_SIZE, TILE_SIZE);
        enc.set_color(png::ColorType::Indexed);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_palette(&palette[..]);
        enc.set_trns(&trns[..]);
        enc.set_compression(png::Compression::Fast);
        let mut w = enc.write_header().unwrap();
        w.write_image_data(coverage).unwrap();
    }
    out
}

/// Repaint a tile produced by [`encode_indexed_coverage`] /
/// [`encode_indexed_coverage_rgb`], returning the same PNG with every palette
/// entry set to `rgb`.
///
/// Coverage lives in the pixel bytes and opacity in `tRNS`, so only `PLTE`
/// changes: 768 bytes plus a fresh CRC, with the compressed `IDAT` copied
/// verbatim. That makes serving a tile in an arbitrary color a memcpy rather
/// than a re-render — the cheap way to answer a per-request `?color=`.
///
/// `None` when the input is not a palette PNG (no `PLTE` before `IDAT`, or a
/// truncated chunk chain), never for a tile this module encoded.
pub fn recolor_indexed_png(png: &[u8], rgb: (u8, u8, u8)) -> Option<Vec<u8>> {
    if png.len() < SIGNATURE.len() || png[..SIGNATURE.len()] != SIGNATURE {
        return None;
    }

    // Walk the chunk chain rather than trusting a fixed offset, so a future
    // encoder that emits an extra chunk ahead of the palette can't corrupt the
    // output. `PLTE` must precede `IDAT`, so reaching the pixels means there is
    // no palette to patch.
    let mut pos = SIGNATURE.len();
    let (data, len) = loop {
        // 4 length + 4 type + 4 CRC per chunk.
        if pos + 12 > png.len() {
            return None;
        }
        let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = &png[pos + 4..pos + 8];
        let data = pos + 8;
        if data + len + 4 > png.len() {
            return None;
        }
        if kind == b"PLTE" {
            break (data, len);
        }
        if kind == b"IDAT" || kind == b"IEND" {
            return None;
        }
        pos = data + len + 4;
    };
    if len == 0 || len % 3 != 0 {
        return None;
    }

    let mut out = png.to_vec();
    let (r, g, b) = rgb;
    for entry in out[data..data + len].as_chunks_mut::<3>().0 {
        entry.copy_from_slice(&[r, g, b]);
    }
    // A chunk's CRC covers its type bytes as well as its data.
    let crc = crc32(&out[data - 4..data + len]);
    out[data + len..data + len + 4].copy_from_slice(&crc.to_be_bytes());
    Some(out)
}

/// PNG's CRC-32 (IEEE, reflected polynomial `0xEDB88320`). Bitwise rather than
/// table-driven: the only input is a 772-byte palette chunk, so a lookup table
/// would cost more cache than it saves, and this keeps the crate dep-free.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> Vec<u8> {
        (0..(TILE_SIZE * TILE_SIZE))
            .map(|i| (i % 251) as u8)
            .collect()
    }

    /// The whole point of the palette trick: repainting an encoded tile has to
    /// land on exactly the bytes a direct encode in that color would produce.
    #[test]
    fn recolor_matches_a_direct_encode() {
        let coverage = ramp();
        let red = (0xFF, 0x3B, 0x30);

        let blue_png = encode_indexed_coverage(&coverage);
        let repainted = recolor_indexed_png(&blue_png, red).unwrap();

        assert_eq!(repainted, encode_indexed_coverage_rgb(&coverage, red));
        // Only the palette moved; the compressed pixels are untouched.
        assert_eq!(repainted.len(), blue_png.len());
    }

    #[test]
    fn recolor_rejects_non_palette_input() {
        assert!(recolor_indexed_png(b"not a png at all", (1, 2, 3)).is_none());
        assert!(recolor_indexed_png(&SIGNATURE, (1, 2, 3)).is_none());
    }

    #[test]
    fn encoded_tile_decodes_to_the_coverage_it_was_given() {
        let coverage = ramp();
        let png_bytes = encode_indexed_coverage(&coverage);

        let mut decoder = png::Decoder::new(std::io::Cursor::new(&png_bytes));
        // Without EXPAND the reader hands back raw palette indices, not the
        // RGBA a map client would see.
        decoder.set_transformations(png::Transformations::EXPAND);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();

        assert_eq!(info.width, TILE_SIZE);
        assert_eq!(info.height, TILE_SIZE);
        assert_eq!(info.color_type, png::ColorType::Rgba);
        // Alpha of the expanded RGBA is the coverage byte we put in.
        let alphas: Vec<u8> = buf[..info.buffer_size()]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| px[3])
            .collect();
        assert_eq!(alphas, coverage);
    }

    #[test]
    fn empty_tile_renders_blank() {
        let r = TileRasterizer::new();
        let empty: Vec<Vec<(f32, f32)>> = Vec::new();
        let jobs = vec![TileJob {
            x: 3,
            y: 4,
            polylines: &empty,
        }];
        let out = r.render_batch(&jobs, 3.0, true, 255);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].x, out[0].y), (3, 4));
        assert!(out[0].coverage.iter().all(|&c| c == 0));
    }
}
