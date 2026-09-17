//! Compact fp16 polyline `.bin` format, all little-endian:
//!
//! ```text
//! per polyline:
//!   u32  point_count
//!   point_count * (f16 x, f16 y)   // tile-local pixel coords in [0, TILE_SIZE]
//! ```
//!
//! Tile coords live in `[0, 512]`. f16 ULP at 256..512 is 0.25 px, well below
//! the stroke widths used at high zoom, so the quantization is invisible.
//!
//! On aarch64 (iOS/Android arm + Apple Silicon) the f16 conversion is the single
//! instruction `FCVT`, part of the ARMv8.0 ASIMD base, so the NEON path needs no
//! extra `target_feature` gate. Other targets use the scalar path.

use half::f16;

/// Encode polylines to the compact fp16 binary form. See module docs for layout.
///
/// Coordinates outside f16's range saturate to infinity; keep points within
/// `[0, TILE_SIZE]` as [`crate::bucketize_track`] produces them.
pub fn encode_bin(polylines: &[Vec<(f32, f32)>]) -> Vec<u8> {
    let total_pts: usize = polylines.iter().map(|p| p.len()).sum();
    let cap = polylines.len() * 4 + total_pts * 4;
    let mut out = Vec::with_capacity(cap);
    for polyline in polylines {
        out.extend_from_slice(&(polyline.len() as u32).to_le_bytes());
        let start = out.len();
        out.resize(start + polyline.len() * 4, 0);
        encode_points(polyline, &mut out[start..]);
    }
    out
}

/// Inverse of [`encode_bin`]. Returns the decoded polylines, or `None` on a
/// malformed buffer (truncated header, truncated payload, or a length prefix
/// larger than the bytes that follow).
///
/// Safe to call on untrusted input: a hostile length prefix allocates nothing
/// before it has been checked against the remaining bytes.
pub fn decode_bin(bytes: &[u8]) -> Option<Vec<Vec<(f32, f32)>>> {
    let mut polylines = Vec::new();
    let mut off = 0usize;
    while off < bytes.len() {
        let count = u32::from_le_bytes(bytes.get(off..off + 4)?.try_into().ok()?) as usize;
        off += 4;
        // Checked: `count` is attacker-controlled and `count * 4` would wrap on
        // a 32-bit target, turning a truncated buffer into an in-bounds read.
        let payload_len = count.checked_mul(4)?;
        let end = off.checked_add(payload_len)?;
        let payload = bytes.get(off..end)?;
        off = end;
        let mut polyline = vec![(0.0f32, 0.0f32); count];
        decode_points(payload, &mut polyline);
        polylines.push(polyline);
    }
    Some(polylines)
}

#[inline]
fn encode_points(pts: &[(f32, f32)], dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), pts.len() * 4);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        encode_points_neon(pts, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    encode_points_scalar(pts, dst);
}

#[inline]
fn decode_points(src: &[u8], dst: &mut [(f32, f32)]) {
    debug_assert_eq!(src.len(), dst.len() * 4);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        decode_points_neon(src, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    decode_points_scalar(src, dst);
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn encode_points_scalar(pts: &[(f32, f32)], dst: &mut [u8]) {
    for (i, &(x, y)) in pts.iter().enumerate() {
        let off = i * 4;
        dst[off..off + 2].copy_from_slice(&f16::from_f32(x).to_le_bytes());
        dst[off + 2..off + 4].copy_from_slice(&f16::from_f32(y).to_le_bytes());
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn decode_points_scalar(src: &[u8], dst: &mut [(f32, f32)]) {
    for (i, slot) in dst.iter_mut().enumerate() {
        let off = i * 4;
        let x = f16::from_le_bytes([src[off], src[off + 1]]);
        let y = f16::from_le_bytes([src[off + 2], src[off + 3]]);
        *slot = (x.to_f32(), y.to_f32());
    }
}

/// # Safety
/// `dst` must be at least `pts.len() * 4` bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_points_neon(pts: &[(f32, f32)], dst: &mut [u8]) {
    use std::arch::aarch64::*;
    use std::mem::transmute;

    let n = pts.len();
    let chunks = n / 4;
    let u16_dst = dst.as_mut_ptr() as *mut u16;

    for c in 0..chunks {
        let base = c * 4;
        // Pack 4 (x,y) pairs into a contiguous f32x8 stack buffer — sidesteps any
        // tuple-layout assumption about `(f32,f32)`.
        let buf: [f32; 8] = unsafe {
            let p0 = *pts.get_unchecked(base);
            let p1 = *pts.get_unchecked(base + 1);
            let p2 = *pts.get_unchecked(base + 2);
            let p3 = *pts.get_unchecked(base + 3);
            [p0.0, p0.1, p1.0, p1.1, p2.0, p2.1, p3.0, p3.1]
        };
        unsafe {
            let v0 = vld1q_f32(buf.as_ptr());
            let v1 = vld1q_f32(buf.as_ptr().add(4));
            let h0 = vcvt_f16_f32(v0);
            let h1 = vcvt_f16_f32(v1);
            // float16x4_t and uint16x4_t are both 64-bit SIMD vectors with identical
            // layout — transmute is sound and avoids depending on the unstable `f16`
            // primitive used by `vst1_f16`.
            let u0: uint16x4_t = transmute(h0);
            let u1: uint16x4_t = transmute(h1);
            vst1_u16(u16_dst.add(base * 2), u0);
            vst1_u16(u16_dst.add(base * 2 + 4), u1);
        }
    }

    for i in (chunks * 4)..n {
        let (x, y) = unsafe { *pts.get_unchecked(i) };
        let off = i * 4;
        dst[off..off + 2].copy_from_slice(&f16::from_f32(x).to_le_bytes());
        dst[off + 2..off + 4].copy_from_slice(&f16::from_f32(y).to_le_bytes());
    }
}

/// # Safety
/// `src` must be at least `dst.len() * 4` bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_points_neon(src: &[u8], dst: &mut [(f32, f32)]) {
    use std::arch::aarch64::*;
    use std::mem::transmute;

    let n = dst.len();
    let chunks = n / 4;
    let u16_src = src.as_ptr() as *const u16;

    for c in 0..chunks {
        let base = c * 4;
        let mut buf = [0.0f32; 8];
        unsafe {
            let u0 = vld1_u16(u16_src.add(base * 2));
            let u1 = vld1_u16(u16_src.add(base * 2 + 4));
            let h0: float16x4_t = transmute(u0);
            let h1: float16x4_t = transmute(u1);
            let v0 = vcvt_f32_f16(h0);
            let v1 = vcvt_f32_f16(h1);
            vst1q_f32(buf.as_mut_ptr(), v0);
            vst1q_f32(buf.as_mut_ptr().add(4), v1);
            *dst.get_unchecked_mut(base) = (buf[0], buf[1]);
            *dst.get_unchecked_mut(base + 1) = (buf[2], buf[3]);
            *dst.get_unchecked_mut(base + 2) = (buf[4], buf[5]);
            *dst.get_unchecked_mut(base + 3) = (buf[6], buf[7]);
        }
    }

    for i in (chunks * 4)..n {
        let off = i * 4;
        let x = f16::from_le_bytes([src[off], src[off + 1]]);
        let y = f16::from_le_bytes([src[off + 2], src[off + 3]]);
        unsafe {
            *dst.get_unchecked_mut(i) = (x.to_f32(), y.to_f32());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let lines = vec![
            vec![(0.0, 0.0), (100.5, 200.25), (511.0, 256.0)],
            vec![(1.0, 2.0), (3.0, 4.0)],
        ];
        let bytes = encode_bin(&lines);
        let back = decode_bin(&bytes).unwrap();
        assert_eq!(back.len(), lines.len());
        for (a, b) in lines.iter().zip(back.iter()) {
            assert_eq!(a.len(), b.len());
            for (&(ax, ay), &(bx, by)) in a.iter().zip(b.iter()) {
                // f16 ULP at 256..512 = 0.25 px.
                assert!((ax - bx).abs() <= 0.25, "x mismatch {ax} vs {bx}");
                assert!((ay - by).abs() <= 0.25, "y mismatch {ay} vs {by}");
            }
        }
    }

    #[test]
    fn roundtrip_tail() {
        // Not a multiple of 4 — exercises the scalar tail of the NEON path.
        let pts: Vec<(f32, f32)> = (0..7).map(|i| (i as f32 * 10.0, i as f32 * 7.5)).collect();
        let lines = vec![pts.clone()];
        let bytes = encode_bin(&lines);
        let back = decode_bin(&bytes).unwrap();
        assert_eq!(back[0].len(), pts.len());
    }

    #[test]
    fn roundtrip_empty_and_single_point_lines() {
        let lines = vec![vec![], vec![(7.0, 9.0)]];
        let bytes = encode_bin(&lines);
        let back = decode_bin(&bytes).unwrap();
        assert_eq!(back.len(), 2);
        assert!(back[0].is_empty());
        assert_eq!(back[1].len(), 1);
    }

    #[test]
    fn decode_rejects_truncated() {
        let mut bytes = (3u32).to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 5]); // 5 < 3 * 4
        assert!(decode_bin(&bytes).is_none());
    }

    #[test]
    fn decode_rejects_a_hostile_length_prefix() {
        // Claims u32::MAX points with no payload: must not allocate or wrap.
        let bytes = u32::MAX.to_le_bytes().to_vec();
        assert!(decode_bin(&bytes).is_none());
    }

    #[test]
    fn decode_rejects_a_truncated_header() {
        assert!(decode_bin(&[1, 2, 3]).is_none());
    }
}
