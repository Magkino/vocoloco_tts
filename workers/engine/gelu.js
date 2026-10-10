// VocoLoco engine: GELU as PyTorch computes it in float32, (x · 0.5) · (1 + erf(x · 0.70710677)),
// for the encoders (HuBERT in audio-encoder.js, Whisper). erf_ is VocoLoco's own approximation:
// piecewise polynomials, no exp (so it doesn't depend on the GPU's exp), fitted by least squares
// on Chebyshev nodes in float64. In float32 it is within 1 ulp of erf on every input and
// correctly rounded on 94.7 % of them: each piece adds its large leading term (x, or erf at the
// piece's centre split into H + L) last, so that add is the only rounding that counts.

export const GELU = `
fn erf_(x: f32) -> f32 {
  let a = abs(x);
  var r = 1.0;
  if (a < 0.75) { let s = a * a; r = a + a * (((((((9.3362636e-05) * s + -0.0008332537) * s + 0.005215088) * s + -0.026864205) * s + 0.112837695) * s + -0.37612638) * s + 0.12837917); }
  else if (a < 1.5) { let t = a - 1.125; r = 0.8883882 + (1.5206194e-08 + t * ((((((((((((-0.00013667744) * t + -0.00015693932) * t + 0.0011524094) * t + -0.0003633144) * t + -0.006031188) * t + 0.00960712) * t + 0.015536833) * t + -0.06132369) * t + 0.027973298) * t + 0.16245234) * t + -0.3580582) * t + 0.31827396)); }
  else if (a < 2.5) { let t = a - 2.0; r = 0.9953223 + (-2.206372e-08 + t * ((((((((((((8.162815e-06) * t + 9.193122e-05) * t + -0.00026369048) * t + -4.6822843e-05) * t + 0.0015911609) * t + -0.003379313) * t + 0.00045925687) * t + 0.013089116) * t + -0.034444977) * t + 0.048222966) * t + -0.04133397) * t + 0.020666985)); }
  else if (a < 3.919206) { let t = a - 3.2096028; r = 0.99999434 + (1.1993277e-08 + t * ((((((((((((2.0764933e-06) * t + -4.546219e-06) * t + -7.5970974e-07) * t + 2.6935948e-05) * t + -9.123229e-05) * t + 0.00019782338) * t + -0.0003154701) * t + 0.00038381846) * t + -0.00035681066) * t + 0.0002476002) * t + -0.000121618476) * t + 3.7892067e-05)); }
  return select(r, -r, x < 0.0);
}
fn gelu1(x: f32) -> f32 { return (x * 0.5) * (1.0 + erf_(x * 0.70710677)); }
fn gelu(y: vec4<f32>) -> vec4<f32> { return vec4<f32>(gelu1(y.x), gelu1(y.y), gelu1(y.z), gelu1(y.w)); }`;
