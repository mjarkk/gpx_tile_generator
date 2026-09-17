//! Render a track into an XYZ tile pyramid.
//!
//! ```sh
//! cargo run --release --example render_track -- ./tiles
//! ```
//!
//! Writes `<out>/{z}/{x}/{y}.png`, servable as-is by any static file server and
//! consumable by Leaflet:
//!
//! ```js
//! L.tileLayer('/tiles/{z}/{x}/{y}.png', { tileSize: 512, maxZoom: 14 }).addTo(map)
//! ```

use std::fs;
use std::path::Path;

use gpx_tile_generator::{
    bucketize_track, encode_indexed_coverage_rgb, project, TileBuckets, TileJob, TileRasterizer,
};

const ZOOMS: std::ops::RangeInclusive<u32> = 9..=14;
const TRACK_COLOR: (u8, u8, u8) = (0x00, 0x7A, 0xFF);

/// A meandering loop, standing in for whatever you parse out of your `.gpx`
/// files — the crate only ever sees `(lat, lon)`.
fn sample_track() -> Vec<(f64, f64)> {
    (0..4000)
        .map(|i| {
            let t = i as f64 / 4000.0 * std::f64::consts::TAU;
            let lat = 0.08 * t.sin() + 0.015 * (t * 7.0).sin();
            let lon = 0.12 * t.cos() + 0.02 * (t * 5.0).cos();
            (lat, lon)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tiles".to_string());
    let out = Path::new(&out);

    let projected: Vec<(f64, f64)> = sample_track()
        .into_iter()
        .map(|(lat, lon)| project(lat, lon))
        .collect();

    let rasterizer = TileRasterizer::new();
    match rasterizer.gpu_info() {
        Some(info) => println!("rasterizing on {info}"),
        None => println!(
            "rasterizing on CPU ({})",
            rasterizer.gpu_error().unwrap_or("no reason given")
        ),
    }

    let mut total = 0usize;
    for zoom in ZOOMS {
        let mut buckets = TileBuckets::default();
        // Thinning scales with zoom: at low zoom many points land on the same
        // pixel, at high zoom every wobble is real detail.
        let min_px2 = if zoom < 12 { 1.0 } else { 0.25 };
        bucketize_track(&projected, (1u32 << zoom) as f64, min_px2, &mut buckets);

        let jobs: Vec<TileJob> = buckets
            .iter()
            .map(|(&(x, y), polylines)| TileJob { x, y, polylines })
            .collect();

        // Thicker strokes as you zoom out keep a sparse track visible.
        let stroke = if zoom < 11 { 2.0 } else { 4.0 };
        let tiles = rasterizer.render_batch(&jobs, stroke, true, 255);

        for tile in &tiles {
            let dir = out.join(zoom.to_string()).join(tile.x.to_string());
            fs::create_dir_all(&dir)?;
            let png = encode_indexed_coverage_rgb(&tile.coverage, TRACK_COLOR);
            fs::write(dir.join(format!("{}.png", tile.y)), png)?;
        }

        println!("z{zoom}: {} tiles", tiles.len());
        total += tiles.len();
    }

    println!("wrote {total} tiles to {}", out.display());
    Ok(())
}
