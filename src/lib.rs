//! Turn GPS tracks into Web Mercator raster map tiles.
//!
//! Given tracks as `(lat, lon)` point sequences, this crate projects them,
//! splits them into per-tile polylines, and strokes those into 512x512 PNG
//! tiles ready to serve to a slippy map (Leaflet, `MapLibre`, XYZ layers).
//!
//! # Pipeline
//!
//! ```text
//! (lat, lon)  --project-->  [0,1]  --bucketize_track-->  per-tile polylines
//!                                                               |
//!                            encode_bin  <--- store ---         |
//!                                                               v
//!                                            TileRasterizer::render_batch
//!                                                               |
//!                                                               v
//!                                            encode_indexed_coverage -> PNG
//! ```
//!
//! - [`project`] — Web Mercator base projection (zoom-independent, `[0, 1]`).
//! - [`bucketize_track`] — split a projected track into per-tile polylines
//!   (maximal in-tile runs), the heart of the per-zoom rasterizer.
//! - [`encode_bin`] / [`decode_bin`] — compact fp16 polyline dump (NEON + scalar).
//! - [`render_tile`] / [`render_tile_indexed_png`] — single-tile CPU stroking.
//! - [`TileRasterizer`] — batched GPU (wgpu) rasterizer with a CPU fallback,
//!   plus [`encode_indexed_coverage`] for single-channel PNG output.
//! - [`recolor_indexed_png`] — repaint an encoded tile via its palette alone,
//!   without re-rendering or re-compressing.
//!
//! Style (color/stroke/AA/opacity) is a parameter ([`TileStyle`]), so the
//! algorithm is shared while the look stays yours.
//!
//! # Example
//!
//! ```
//! use gpx_tile_generator::{bucketize_track, project, render_tile_indexed_png, TileStyle};
//! use gpx_tile_generator::FxHashMap;
//!
//! let track: Vec<(f64, f64)> = vec![(0.0, 0.0), (0.01, 0.02), (0.03, 0.05)]
//!     .into_iter()
//!     .map(|(lat, lon)| project(lat, lon))
//!     .collect();
//!
//! let zoom = 14u32;
//! let mut buckets = FxHashMap::default();
//! bucketize_track(&track, (1u32 << zoom) as f64, 0.25, &mut buckets);
//!
//! let style = TileStyle { color_rgba: (0, 122, 255, 255), stroke_width: 3.0, anti_alias: true };
//! for ((x, y), polylines) in &buckets {
//!     if let Some(png) = render_tile_indexed_png(polylines, &style) {
//!         // write to tiles/{zoom}/{x}/{y}.png
//!         let _ = (x, y, png);
//!     }
//! }
//! ```
//!
//! # GPU acceleration
//!
//! [`TileRasterizer`] uses wgpu (Metal / Vulkan / DX12) when a device is
//! available and transparently falls back to CPU rasterization otherwise. On
//! `wasm32` the GPU path is compiled out entirely.
//!
//! # Cargo features
//!
//! - `gpu` *(default)* — the wgpu batched rasterizer. Disable for a
//!   dependency-light CPU-only build; [`TileRasterizer`] keeps its API and
//!   always takes the CPU path.

#![warn(missing_docs)]
#![warn(clippy::doc_markdown)]
#![forbid(unsafe_op_in_unsafe_fn)]

use tiny_skia::{LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, Transform};

mod bin_codec;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod gpu;
mod raster;

pub use bin_codec::{decode_bin, encode_bin};
pub use raster::{
    encode_indexed_coverage, encode_indexed_coverage_rgb, recolor_indexed_png, RenderedTile,
    TileJob, TileRasterizer,
};
/// Re-exported: [`bucketize_track`] fills one of these, so callers need the
/// exact `rustc-hash` version this crate was built against.
pub use rustc_hash::FxHashMap;

/// One polyline in tile-local pixel coords, `(0, 0)` at the tile's top-left.
pub type Polyline = Vec<(f32, f32)>;

/// Polylines grouped by `(tile_x, tile_y)` at one zoom level — what
/// [`bucketize_track`] accumulates into and what the renderers consume.
pub type TileBuckets = FxHashMap<(u32, u32), Vec<Polyline>>;

/// Side length (px) of every rendered tile. Oversampled vs the 256 dp logical
/// map tile so retina screens stay crisp.
///
/// Fixed at compile time: the GPU path packs tiles into a 8192² meta texture
/// sized around it, and the `.bin` codec's fp16 coordinates are chosen for this
/// range. Serving these as 256 dp tiles is a matter of the client's `tileSize`.
pub const TILE_SIZE: u32 = 512;

/// Web Mercator base projection — output coords in `[0, 1]`, zoom independent.
/// Multiply by `n = 2^zoom` to get per-zoom tile coords.
///
/// `lat` is clamped by the projection itself only insofar as the Mercator
/// formula diverges at the poles; latitudes beyond ±85.051129° project outside
/// `[0, 1]` and are the caller's to filter.
#[inline]
pub fn project(lat: f64, lon: f64) -> (f64, f64) {
    let lat_rad = lat.to_radians();
    let base_x = (lon + 180.0) / 360.0;
    let base_y = (1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0;
    (base_x, base_y)
}

/// Per-tile stroke style. Color is straight (non-premultiplied) RGBA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileStyle {
    /// Stroke color, straight (non-premultiplied) RGBA.
    pub color_rgba: (u8, u8, u8, u8),
    /// Stroke width in tile pixels (the tile is [`TILE_SIZE`] wide).
    pub stroke_width: f32,
    /// Anti-alias stroke edges.
    pub anti_alias: bool,
}

/// Stroke the given tile-local ([`TILE_SIZE`]² pixel space) polylines onto a
/// fresh pixmap. Anything outside the tile is clipped.
///
/// A single `PathBuilder` accumulates every polyline so overlapping strokes
/// union (one `stroke_path` call) instead of compositing additively — 10
/// overlapping passes over the same road look identical to 1.
pub fn render_tile(polylines: &[Vec<(f32, f32)>], style: &TileStyle) -> Pixmap {
    let mut pixmap = Pixmap::new(TILE_SIZE, TILE_SIZE).unwrap();

    let mut paint = Paint::default();
    let (r, g, b, a) = style.color_rgba;
    paint.set_color_rgba8(r, g, b, a);
    paint.anti_alias = style.anti_alias;

    let mut pb = PathBuilder::new();
    for polyline in polylines {
        let mut pts = polyline.iter();
        if let Some(&(sx, sy)) = pts.next() {
            pb.move_to(sx, sy);
            for &(px, py) in pts {
                pb.line_to(px, py);
            }
        }
    }

    if let Some(path) = pb.finish() {
        let stroke = Stroke {
            width: style.stroke_width,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            ..Default::default()
        };
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }

    pixmap
}

/// Render polylines and encode the result as an indexed-coverage PNG (the same
/// compact format the batched rasterizer emits), in `style`'s color.
///
/// Returns `None` if `polylines` is empty, so a tile with no geometry is
/// distinguishable from a blank one. Output is byte-identical in look to a tile
/// [`TileRasterizer`] produced with the same style, which is what makes
/// on-demand rendering of sparse tiles mix with pre-rendered dense ones.
pub fn render_tile_indexed_png(
    polylines: &[Vec<(f32, f32)>],
    style: &TileStyle,
) -> Option<Vec<u8>> {
    if polylines.is_empty() {
        return None;
    }
    let pixmap = render_tile(polylines, style);
    let rgba = pixmap.take();
    // Coverage = alpha channel of the stroked tile (opacity already baked in
    // by the paint alpha), exactly what the GPU path produces. The RGB the
    // pixmap carries is discarded and re-applied as the PNG palette.
    let coverage: Vec<u8> = (0..(TILE_SIZE * TILE_SIZE) as usize)
        .map(|i| rgba[i * 4 + 3])
        .collect();
    let (r, g, b, _) = style.color_rgba;
    Some(encode_indexed_coverage_rgb(&coverage, (r, g, b)))
}

/// Split one projected track (base coords in `[0, 1]`, as [`project`] returns)
/// at zoom `n = 2^zoom` into per-tile polylines, appending each maximal in-tile
/// run to `buckets` keyed by `(tile_x, tile_y)`.
///
/// When a segment crosses to a new tile the current run is closed with the
/// crossing endpoint (projected in the OLD tile's coords) and flushed, then a
/// new run is seeded with `[prev, cur]` so the crossing draws in BOTH tiles
/// (the per-tile clip drops the off-tile part). Without this, strokes would
/// visibly break at every tile seam.
///
/// `min_px2` drops a candidate point whose squared pixel distance from the last
/// kept point is below the threshold (no visible detail, just stroke work).
/// Boundary crossings and the first point of each run are always kept. Pass
/// `0.0` to keep every point.
///
/// Tracks shorter than 2 points are ignored. Points projecting to negative tile
/// indices (latitudes past the Mercator limit) are dropped rather than panicking.
///
/// `buckets` is appended to, never cleared, so many tracks accumulate into one
/// map by calling this repeatedly.
pub fn bucketize_track(track: &[(f64, f64)], n: f64, min_px2: f32, buckets: &mut TileBuckets) {
    if track.len() < 2 {
        return;
    }

    let project = |p: (f64, f64), xb: f64, yb: f64| -> (f32, f32) {
        (
            ((p.0 * n - xb) * TILE_SIZE as f64) as f32,
            ((p.1 * n - yb) * TILE_SIZE as f64) as f32,
        )
    };

    let mut prev = track[0];
    let (mut tile_xb, mut tile_yb) = ((prev.0 * n).floor(), (prev.1 * n).floor());
    let mut run: Vec<(f32, f32)> = vec![project(prev, tile_xb, tile_yb)];

    for &cur in &track[1..] {
        let cur_xb = (cur.0 * n).floor();
        let cur_yb = (cur.1 * n).floor();

        if cur_xb == tile_xb && cur_yb == tile_yb {
            let p = project(cur, tile_xb, tile_yb);
            let last = *run.last().unwrap();
            let (dx, dy) = (p.0 - last.0, p.1 - last.1);
            if dx * dx + dy * dy >= min_px2 {
                run.push(p);
            }
            prev = cur;
            continue;
        } else {
            run.push(project(cur, tile_xb, tile_yb));
            // Negative tile indices can't happen for tracks projected from
            // valid lat/lon (base coords in [0, 1] => floor(base * n) >= 0),
            // but guard anyway so a stray point can't panic the cast.
            if tile_xb >= 0.0 && tile_yb >= 0.0 {
                buckets
                    .entry((tile_xb as u32, tile_yb as u32))
                    .or_default()
                    .push(std::mem::take(&mut run));
            } else {
                run.clear();
            }

            tile_xb = cur_xb;
            tile_yb = cur_yb;
            run = vec![
                project(prev, tile_xb, tile_yb),
                project(cur, tile_xb, tile_yb),
            ];
        }

        prev = cur;
    }

    if run.len() > 1 && tile_xb >= 0.0 && tile_yb >= 0.0 {
        buckets
            .entry((tile_xb as u32, tile_yb as u32))
            .or_default()
            .push(run);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_origin() {
        let (x, y) = project(0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-9);
        assert!((y - 0.5).abs() < 1e-9);
    }

    #[test]
    fn project_orients_east_and_north() {
        let (x0, y0) = project(10.0, -20.0);
        let (x1, y1) = project(20.0, -10.0);
        assert!(x1 > x0, "east increases x");
        assert!(y1 < y0, "north decreases y");
    }

    #[test]
    fn bucketize_single_tile_run() {
        // A short track entirely inside tile (0,0) at zoom 0 (n = 1).
        let track = vec![(0.10, 0.10), (0.11, 0.11), (0.12, 0.12)];
        let mut buckets = rustc_hash::FxHashMap::default();
        bucketize_track(&track, 1.0, 1.0, &mut buckets);
        assert_eq!(buckets.len(), 1);
        let polys = &buckets[&(0, 0)];
        assert_eq!(polys.len(), 1);
        assert!(polys[0].len() >= 2);
    }

    #[test]
    fn bucketize_crossing_draws_in_both_tiles() {
        // Crosses the x=0.5 seam at zoom 1 (n = 2), so tiles (0,0) and (1,0).
        let track = vec![(0.45, 0.25), (0.55, 0.25)];
        let mut buckets = rustc_hash::FxHashMap::default();
        bucketize_track(&track, 2.0, 0.0, &mut buckets);
        assert!(buckets.contains_key(&(0, 0)));
        assert!(buckets.contains_key(&(1, 0)));
    }

    #[test]
    fn bucketize_ignores_degenerate_tracks() {
        let mut buckets = rustc_hash::FxHashMap::default();
        bucketize_track(&[], 1.0, 0.0, &mut buckets);
        bucketize_track(&[(0.5, 0.5)], 1.0, 0.0, &mut buckets);
        assert!(buckets.is_empty());
    }

    #[test]
    fn render_tile_indexed_png_is_none_when_empty() {
        let style = TileStyle {
            color_rgba: (0, 0, 0, 255),
            stroke_width: 2.0,
            anti_alias: true,
        };
        assert!(render_tile_indexed_png(&[], &style).is_none());
    }
}
