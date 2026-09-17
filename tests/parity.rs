//! The GPU and CPU rasterizers must be interchangeable: a tile pre-rendered in
//! bulk on the GPU sits next to one rendered on demand on the CPU, and a seam
//! between them would be visible on the map.

use gpx_tile_generator::{
    bucketize_track, decode_bin, encode_bin, encode_indexed_coverage, project, recolor_indexed_png,
    render_tile, TileBuckets, TileJob, TileRasterizer, TileStyle, TILE_SIZE,
};

/// A handful of distinct polylines, enough tiles to clear the rasterizer's
/// small-batch CPU fallback.
fn jobs_geometry(count: usize) -> Vec<Vec<Vec<(f32, f32)>>> {
    (0..count)
        .map(|i| {
            let o = (i * 17 % 200) as f32;
            vec![
                vec![(10.0 + o, 10.0), (500.0, 300.0 + o), (250.0, 500.0)],
                vec![(0.0, 256.0 + o), (511.0, 256.0 - o)],
                // Crosses the tile edge: exercises the per-cell clip.
                vec![(-40.0, 100.0), (600.0, 120.0)],
            ]
        })
        .collect()
}

fn cpu_coverage(polylines: &[Vec<(f32, f32)>], stroke: f32, aa: bool, alpha: u8) -> Vec<u8> {
    let style = TileStyle {
        color_rgba: (0, 0, 0, alpha),
        stroke_width: stroke,
        anti_alias: aa,
    };
    let rgba = render_tile(polylines, &style).take();
    (0..(TILE_SIZE * TILE_SIZE) as usize)
        .map(|i| rgba[i * 4 + 3])
        .collect()
}

#[test]
fn batch_matches_single_tile_rendering() {
    let geom = jobs_geometry(8);
    let jobs: Vec<TileJob> = geom
        .iter()
        .enumerate()
        .map(|(i, polylines)| TileJob {
            x: i as u32,
            y: 0,
            polylines,
        })
        .collect();

    let stroke = 4.0;
    let alpha = 255u8;
    let rasterizer = TileRasterizer::new();
    let rendered = rasterizer.render_batch(&jobs, stroke, true, alpha);

    assert_eq!(rendered.len(), jobs.len());

    for (tile, polylines) in rendered.iter().zip(geom.iter()) {
        let reference = cpu_coverage(polylines, stroke, true, alpha);
        assert_eq!(tile.coverage.len(), reference.len());

        let painted = reference.iter().filter(|&&c| c > 0).count();
        assert!(painted > 1000, "test geometry should cover real area");

        let total_diff: u64 = tile
            .coverage
            .iter()
            .zip(reference.iter())
            .map(|(&a, &b)| a.abs_diff(b) as u64)
            .sum();
        let mean_diff = total_diff as f64 / reference.len() as f64;

        // The two rasterizers use different AA maths (analytic SDF vs scanline
        // coverage), so edge pixels differ; a visible mismatch would move the
        // mean far past this.
        assert!(
            mean_diff < 1.0,
            "GPU/CPU coverage diverged (mean {mean_diff:.3}/255, gpu={})",
            rasterizer.is_gpu()
        );
    }
}

#[test]
fn batch_preserves_input_order_and_keys() {
    let geom = jobs_geometry(8);
    let jobs: Vec<TileJob> = geom
        .iter()
        .enumerate()
        .map(|(i, polylines)| TileJob {
            x: 100 + i as u32,
            y: 200 + i as u32,
            polylines,
        })
        .collect();

    let rendered = TileRasterizer::new().render_batch(&jobs, 3.0, true, 255);
    let keys: Vec<(u32, u32)> = rendered.iter().map(|t| (t.x, t.y)).collect();
    let expected: Vec<(u32, u32)> = jobs.iter().map(|j| (j.x, j.y)).collect();
    assert_eq!(keys, expected);
}

#[test]
fn strokes_do_not_bleed_between_tiles() {
    // One tile whose only geometry lies entirely to the left of it. Nothing may
    // land inside; if the GPU cell clip regressed, this fills with coverage.
    let far_left = vec![vec![(-800.0, 100.0), (-600.0, 400.0)]];
    let jobs: Vec<TileJob> = (0..8)
        .map(|i| TileJob {
            x: i,
            y: 0,
            polylines: &far_left,
        })
        .collect();

    for tile in TileRasterizer::new().render_batch(&jobs, 6.0, true, 255) {
        assert!(tile.coverage.iter().all(|&c| c == 0));
    }
}

#[test]
fn full_pipeline_track_to_recolored_png() {
    let track: Vec<(f64, f64)> = (0..400)
        .map(|i| {
            let t = i as f64 / 400.0;
            project(t * 0.2, t * 0.3)
        })
        .collect();

    let zoom = 12u32;
    let mut buckets = TileBuckets::default();
    bucketize_track(&track, (1u32 << zoom) as f64, 0.25, &mut buckets);
    assert!(buckets.len() > 1, "a 0.2° track must span several tiles");

    let jobs: Vec<TileJob> = buckets
        .iter()
        .map(|(&(x, y), polylines)| TileJob { x, y, polylines })
        .collect();

    let rendered = TileRasterizer::new().render_batch(&jobs, 3.0, true, 255);
    assert_eq!(rendered.len(), jobs.len());

    let with_ink = rendered
        .iter()
        .filter(|t| t.coverage.iter().any(|&c| c > 0))
        .count();
    assert_eq!(with_ink, rendered.len(), "every bucketed tile has geometry");

    let png = encode_indexed_coverage(&rendered[0].coverage);
    let red = recolor_indexed_png(&png, (0xFF, 0x00, 0x00)).unwrap();
    assert_eq!(red.len(), png.len());
    assert_ne!(red, png);
}

#[test]
fn bin_roundtrip_survives_a_real_bucketed_track() {
    let track: Vec<(f64, f64)> = (0..200)
        .map(|i| {
            let t = i as f64 / 200.0;
            project(t * 0.05, t * 0.05)
        })
        .collect();

    let mut buckets = TileBuckets::default();
    bucketize_track(&track, (1u32 << 14) as f64, 0.0, &mut buckets);

    for polylines in buckets.values() {
        let decoded = decode_bin(&encode_bin(polylines)).unwrap();
        assert_eq!(decoded.len(), polylines.len());
        for (orig, back) in polylines.iter().zip(decoded.iter()) {
            assert_eq!(orig.len(), back.len());
            for (&(ax, ay), &(bx, by)) in orig.iter().zip(back.iter()) {
                assert!((ax - bx).abs() <= 0.25);
                assert!((ay - by).abs() <= 0.25);
            }
        }
    }
}
