use std::simd::{simd_swizzle, u64x2, u8x16, u16x8};


// https://github.com/hcs0/Hackers-Delight/blob/master/transpose8.c.txt
#[inline(never)]
pub fn transpose_u64(x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AA;
    let x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC;
    let x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0;
    let x = x ^ t ^ (t << 28);
    x
}

#[inline(always)]
pub fn transpose_u64x2(x: u64x2) -> u64x2 {
    let shift_1 = u64x2::splat(7);
    let shift_2 = u64x2::splat(14);
    let shift_3 = u64x2::splat(28);

    let mask_1 = u64x2::splat(0x00AA00AA00AA00AA);
    let mask_2 = u64x2::splat(0x0000CCCC0000CCCC);
    let mask_3 = u64x2::splat(0x00000000F0F0F0F0);

    let t = (x ^ (x >> shift_1)) & mask_1;
    let x = x ^ t ^ (t << shift_1);

    let t = (x ^ (x >> shift_2)) & mask_2;
    let x = x ^ t ^ (t << shift_2);

    let t = (x ^ (x >> shift_3)) & mask_3;
    let x = x ^ t ^ (t << shift_3);

    x
}

#[inline(always)]
pub fn transpose_u16x8(x: u16x8) -> u8x16 {
    use std::mem;

    let x = unsafe { mem::transmute::<_, u8x16>(x) };
    let x = simd_swizzle!(x, [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]);

    let x = unsafe { mem::transmute::<_, u64x2>(x) };
    let x = transpose_u64x2(x);
    unsafe { mem::transmute::<_, u8x16>(x) }



    // // Safe to transmute due to same alignment?
    // let x = unsafe { mem::transmute::<_, u64x2>(x) };
    // let x = transpose_u64x2(x);
    // let x = unsafe { mem::transmute::<_, u8x16>(x) };
    // simd_swizzle!(x, [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15])
}

#[inline(always)]
pub fn match_table_simd(x: [u8; 8], key_len: usize) -> u8x16 {
    debug_assert!(x.iter().all(|c| *c < 16));

    let mut buf = [0u16; 8];
    for i in 0..8 {
        buf[i] = x[i] as u16;
    }
    let x = u16x8::from(buf);
    let x = u16x8::splat(1) << x;
    let match_table = transpose_u16x8(x);

    // Mask off indexes past our end.
    let mask = !(0b1111_1111 << key_len);
    u8x16::splat(mask) & match_table
}

fn match_table_naive(x: [u8; 8], key_len: usize) -> u8x16 {
    let mut out = [0u8; 16];
    for i in 0..8 {
        if i < key_len {
            out[x[i] as usize] |= 1 << i;
        }
    }
    out.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_transpose() -> anyhow::Result<()> {
        let x: u64 = std::env::var("INPUT")?.parse()?;
        println!("{} -> {:?}", x, transpose_u64x2([x, x].into()));

        let mut x = [0u16; 8];
        for i in 0..8 {
            x[i] = 1 << 0;
        }
        let x = u16x8::from(x);
        println!("{:?} -> {:?}", x, transpose_u16x8(x));
        Ok(())
    }

    fn input_fragment() -> impl Strategy<Value = ([u8; 8], usize)> {
        prop::collection::vec(0u8..16, 0..8)
            .prop_map(|v| {
                let mut out = [0u8; 8];
                for (i, b) in v.iter().enumerate() {
                    out[i] = *b;
                }
                (out, v.len())
            })
    }

    // #[test]
    // fn test_match_table() {
    //     let x = [0; 8];
    //     assert_eq!(match_table_naive(x), match_table_simd(x));
    // }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 8388608, failure_persistence: None, .. ProptestConfig::default() })]

        #[test]
        fn test_match_table((x, key_len) in input_fragment()) {
            assert_eq!(match_table_naive(x, key_len), match_table_simd(x, key_len));
        }
    }
}
