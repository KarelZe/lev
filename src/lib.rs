use pyo3::prelude::*;

/// Calculates the Levenshtein distance between two strings.
#[pyfunction]
#[allow(unused_variables)]
fn distance(s_1: String, s_2: String) -> PyResult<i32> {
    Ok(0_i32)
}

/// Calculates the Levenshtein ratio between two strings.
///
/// It is calculated as `1 - (distance / (len1 + len2))`.
#[pyfunction]
#[allow(unused_variables)]
fn ratio(s_1: String, s_2: String) -> PyResult<f32> {
    Ok(1.0_f32)
}

/// A Python module implemented in Rust for the Levenshtein distance.
#[pymodule]
fn lev(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    Ok(())
}
