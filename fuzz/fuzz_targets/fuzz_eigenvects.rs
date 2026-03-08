#![no_main]

use libfuzzer_sys::fuzz_target;
use symplex::prelude::*;

fuzz_target!(|data: &[u8]| {
    // Need at least 4 bytes for a 2x2 matrix of small integers
    if data.len() < 4 {
        return;
    }

    // Determine matrix size: 2x2 or 3x3 based on available data
    let (n, entries) = if data.len() >= 9 {
        // 3x3 matrix from 9 bytes
        (3, &data[..9])
    } else {
        // 2x2 matrix from 4 bytes
        (2, &data[..4])
    };

    // Build matrix from bytes as small integers in [-10, 10]
    let var = symplex::var("fuzz_lambda");
    let rows: Vec<Vec<Ex>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let byte = entries[i * n + j];
                    // Map 0..255 to -10..10
                    let val = (byte as i64 % 21) - 10;
                    symplex::int(val)
                })
                .collect()
        })
        .collect();

    let m = match Matrix::new(rows) {
        Ok(mat) => mat,
        Err(_) => return,
    };

    // eigenvects() must never panic — it returns Result
    let eigvs = match m.eigenvects(&var) {
        Ok(ev) => ev,
        Err(_) => return, // solver couldn't find eigenvalues — that's fine
    };

    // For each returned eigenvector, verify A·v = λ·v numerically
    for (eigenval, _mult, vecs) in &eigvs {
        // Try to evaluate eigenvalue to f64
        let lambda_f64 = match eigenval.eval_f64() {
            Ok(v) => v,
            Err(_) => continue, // complex or symbolic eigenvalue — skip check
        };

        for vec in vecs {
            // Compute A·v
            let av = match m.matmul(vec) {
                Ok(result) => result,
                Err(_) => continue,
            };

            // Compute λ·v
            let lambda_v = vec.scale(eigenval);

            // Compare element-wise numerically
            for i in 0..n {
                let av_i = match av.get(i, 0).eval_f64() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let lv_i = match lambda_v.get(i, 0).eval_f64() {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Allow tolerance for floating-point comparison
                let diff = (av_i - lv_i).abs();
                let scale = av_i.abs().max(lv_i.abs()).max(1e-15);
                assert!(
                    diff / scale < 1e-6,
                    "A·v ≠ λ·v: eigenvalue={}, row={}, A·v[i]={}, λ·v[i]={}, diff={}",
                    lambda_f64,
                    i,
                    av_i,
                    lv_i,
                    diff
                );
            }
        }
    }
});
