"""GPU renderer for the skinned uniform mesh (mesh renderer "gpu"): OpenGL 4.3 through moderngl, on the graphics card.

Same inputs and outputs as render_mesh (the CPU renderer): `render` -> (rgb uint8 HxWx3, instance mask bool HxW), `render_layer` ->
RGBA. The skinning is the same numpy code (render_mesh.skin_mesh), so the geometry - and the ground truth that comes from it - is the
same; only the picture is better. Passes, all at SUPERSAMPLE x the output size:
  1. sun shadow map (orthographic, 4096^2, fitted to the mesh);
  2. geometry pre-pass: camera-frame position + normal + instance id (no MSAA) -> the instance mask (coverage >= 0.5 of the samples,
     the same rule as the CPU renderer) and the input of
  3. screen-space ambient occlusion (hemisphere kernel, depth-aware blur);
  4. shading, 4x MSAA: glTF metal/roughness PBR (GGX, Smith, Schlick) with the model's base colour, normal and metal/roughness maps
     (mip-mapped, 16x anisotropic), a sun with soft (Poisson PCF) shadows, image-based light from a sky/ground environment tinted by
     the backplate (irradiance as 9 spherical harmonics, prefiltered specular per roughness), ACES tone curve, sRGB;
  5. ground shadow: the shadow map on the ground plane z = 0, darkening the backplate under the officer (not part of the mask).
Then the samples are averaged down to the output size and composited over the backplate.

Not physically exact (no global illumination between the officer and the street, no cloth simulation) and not the Unreal backend.

Contents:
  constants: SUPERSAMPLE, MSAA, SHADOW_SIZE, GROUND_SHADOW
  available — (True, renderer name) when an OpenGL 4.3 context can be created, else (False, reason)
  render — Returns (rgb uint8 HxWx3, mask bool HxW)
  render_layer — The officer alone as an RGBA uint8 image
  environment — Sky/ground environment (equirect, linear RGB) for a backplate
  (+ private helpers: the GL worker thread, shaders, IBL precomputation)
"""
from __future__ import annotations

import queue
import threading

import numpy as np
from PIL import Image

from .camera import Camera
from .render_mesh import _tex_of, material_albedo, skin_mesh, tangent_frame, _blend_dirs
from .skeleton import FKResult
from .uniform_mesh import UniformMesh

SUPERSAMPLE = 2
MSAA = 4
SHADOW_SIZE = 4096
GROUND_SHADOW = 0.6          # how dark the officer's shadow on the ground is at full sun (0 = no ground shadow)
DEBUG = 0
SKY_BASE = (0.74, 0.75, 0.77)
TONE_MAP = "aces"        # "neutral" (Khronos PBR Neutral: true base colours, like a model viewer) or "aces" (filmic)
SUN_GAIN = 0.9          # sun irradiance per unit of lighting.sun_intensity
ENV_GAIN = 1.4          # environment radiance per unit of lighting.ambient_intensity
MAX_CANVAS_SIDE = 8192       # supersampled framebuffer side limit (GL_MAX_RENDERBUFFER_SIZE is larger, memory is the limit)
ENV_W, ENV_H = 64, 32          # the sky/ground environment is smooth: 64 x 32 is plenty
SPEC_ROUGHNESS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# ----------------------------------------------------------------------------- shaders
_GEO_VS = """
#version 430
uniform mat3 R; uniform vec3 C; uniform vec4 proj; uniform vec2 size; uniform float near; uniform float far;
in vec3 in_pos; in vec3 in_nrm;
out vec3 v_pc; out vec3 v_nc;
vec4 clip_of(vec3 pc) {            // pc = (depth, right, up) in the camera frame, cm
    float d = pc.x;
    float x = (2.0 * proj.x / size.x) * pc.y + (2.0 * proj.z / size.x - 1.0) * d;
    float y = (2.0 * proj.y / size.y) * pc.z + (1.0 - 2.0 * proj.w / size.y) * d;
    float z = d * (far + near) / (far - near) - 2.0 * far * near / (far - near);
    return vec4(x, y, z, d);
}
void main() {
    v_pc = transpose(R) * (in_pos - C);
    v_nc = transpose(R) * in_nrm;
    gl_Position = clip_of(v_pc);
}
"""
_GEO_FS = """
#version 430
uniform float inst;
in vec3 v_pc; in vec3 v_nc;
layout(location = 0) out vec4 o_pos;
layout(location = 1) out vec4 o_nrm;
void main() {
    vec3 n = normalize(v_nc);
    if (!gl_FrontFacing) n = -n;
    o_pos = vec4(v_pc, inst);
    o_nrm = vec4(n, 1.0);
}
"""
_SHADOW_VS = """
#version 430
uniform mat4 L;
in vec3 in_pos;
void main() { gl_Position = L * vec4(in_pos, 1.0); }
"""
_SHADOW_FS = """
#version 430
void main() {}
"""
_QUAD_VS = """
#version 430
in vec2 in_xy;
out vec2 v_uv;
void main() { v_uv = in_xy * 0.5 + 0.5; gl_Position = vec4(in_xy, 0.0, 1.0); }
"""
_SSAO_FS = """
#version 430
uniform sampler2D posTex; uniform sampler2D nrmTex;
uniform vec4 proj; uniform vec2 size; uniform float radius; uniform vec3 kernel[32]; uniform int nk;
out float o_ao;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 P = texelFetch(posTex, px, 0);
    if (P.w <= 0.0) { o_ao = 1.0; return; }
    vec3 N = normalize(texelFetch(nrmTex, px, 0).xyz);
    float a = hash(gl_FragCoord.xy) * 6.2831853;
    vec3 rv = vec3(0.0, cos(a), sin(a));
    vec3 T = normalize(rv - N * dot(rv, N));
    if (any(isnan(T)) || length(rv - N * dot(rv, N)) < 1e-3) T = normalize(cross(N, vec3(1.0, 0.0, 0.0)));
    vec3 B = cross(N, T);
    float occ = 0.0;
    for (int i = 0; i < nk; i++) {
        vec3 k = kernel[i];
        vec3 S = P.xyz + (T * k.x + B * k.y + N * k.z) * radius;
        if (S.x < 1.0) continue;
        vec2 uv = vec2(proj.z + proj.x * S.y / S.x, size.y - (proj.w - proj.y * S.z / S.x));
        ivec2 q = ivec2(uv);
        if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y)) continue;
        vec4 Q = texelFetch(posTex, q, 0);
        if (Q.w <= 0.0) continue;
        float range = smoothstep(0.0, 1.0, radius / max(abs(P.x - Q.x), 1e-3));
        occ += (Q.x < S.x - 0.4 ? 1.0 : 0.0) * range;
    }
    o_ao = 1.0 - occ / float(nk);
}
"""
_BLUR_FS = """
#version 430
uniform sampler2D aoTex; uniform sampler2D posTex;
out float o_ao;
void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 P = texelFetch(posTex, px, 0);
    if (P.w <= 0.0) { o_ao = 1.0; return; }
    float s = 0.0, w = 0.0;
    ivec2 sz = textureSize(aoTex, 0);
    for (int y = -3; y <= 3; y++) for (int x = -3; x <= 3; x++) {
        ivec2 q = clamp(px + ivec2(x, y), ivec2(0), sz - 1);
        vec4 Q = texelFetch(posTex, q, 0);
        if (Q.w <= 0.0) continue;
        float wd = exp(-abs(Q.x - P.x) * 0.5);
        s += texelFetch(aoTex, q, 0).r * wd; w += wd;
    }
    o_ao = w > 0.0 ? s / w : 1.0;
}
"""
_SHADE_VS = """
#version 430
uniform mat3 R; uniform vec3 C; uniform vec4 proj; uniform vec2 size; uniform float near; uniform float far;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_tan; in vec3 in_bit; in vec2 in_uv; in float in_jit;
out vec3 v_pw; out vec3 v_n; out vec3 v_t; out vec3 v_b; out vec2 v_uv; out float v_jit;
void main() {
    vec3 pc = transpose(R) * (in_pos - C);
    float d = pc.x;
    float x = (2.0 * proj.x / size.x) * pc.y + (2.0 * proj.z / size.x - 1.0) * d;
    float y = (2.0 * proj.y / size.y) * pc.z + (1.0 - 2.0 * proj.w / size.y) * d;
    float z = d * (far + near) / (far - near) - 2.0 * far * near / (far - near);
    gl_Position = vec4(x, y, z, d);
    v_pw = in_pos; v_n = in_nrm; v_t = in_tan; v_b = in_bit; v_uv = in_uv; v_jit = in_jit;
}
"""
_SHADE_FS = """
#version 430
uniform sampler2D baseTex; uniform sampler2D normalTex; uniform sampler2D mrTex;
uniform int hasBase; uniform int hasNormal; uniform int hasMR;
uniform vec3 baseColor; uniform float roughDefault; uniform float metalDefault; uniform float brightness;
uniform sampler2D aoTex; uniform sampler2DShadow shadowTex; uniform mat4 L; uniform float shadowTexel;
uniform vec3 camPos; uniform vec3 sunDir; uniform vec3 sunColor;
uniform vec3 sh[9]; uniform sampler2DArray specEnv; uniform int specLayers; uniform float envScale;
uniform float exposure; uniform int debugMode; uniform int toneMap;
in vec3 v_pw; in vec3 v_n; in vec3 v_t; in vec3 v_b; in vec2 v_uv; in float v_jit;
out vec4 o_col;
const float PI = 3.14159265;
vec3 srgb2lin(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
vec3 lin2srgb(vec3 c) { c = clamp(c, 0.0, 1.0); return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
vec3 aces(vec3 c) {                  // ACES filmic as three.js has it (Hill's RRT+ODT fit in AP1): the look of web model viewers such as Tripo's
    const mat3 inM = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
    const mat3 outM = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
    vec3 v = inM * (c / 0.6);
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(outM * (a / b), 0.0, 1.0);
}
vec3 neutral(vec3 c) {               // Khronos PBR Neutral tone mapper: keeps the hue and saturation of the base colour (like a model viewer)
    const float start = 0.76, desat = 0.15;
    float x = min(c.r, min(c.g, c.b));
    float off = x < 0.08 ? x - 6.25 * x * x : 0.04;
    c -= off;
    float peak = max(c.r, max(c.g, c.b));
    if (peak < start) return c;
    float d = 1.0 - start;
    float np = 1.0 - d * d / (peak + d - start);
    c *= np / peak;
    float g = 1.0 - 1.0 / (desat * (peak - np) + 1.0);
    return mix(c, vec3(np), g);
}
vec3 irradiance(vec3 n) {           // 9 SH coefficients of the environment radiance -> irradiance / PI
    const float c1 = 0.429043, c2 = 0.511664, c3 = 0.743125, c4 = 0.886227, c5 = 0.247708;
    vec3 e = c1 * sh[8] * (n.x * n.x - n.y * n.y) + c3 * sh[6] * n.z * n.z + c4 * sh[0] - c5 * sh[6]
           + 2.0 * c1 * (sh[4] * n.x * n.y + sh[7] * n.x * n.z + sh[5] * n.y * n.z)
           + 2.0 * c2 * (sh[3] * n.x + sh[1] * n.y + sh[2] * n.z);
    return max(e, vec3(0.0)) / PI;
}
vec3 envSpec(vec3 r, float rough) {
    vec2 uv = vec2(atan(r.y, r.x) / (2.0 * PI) + 0.5, acos(clamp(r.z, -1.0, 1.0)) / PI);
    float l = rough * float(specLayers - 1);
    float l0 = floor(l);
    vec3 a = texture(specEnv, vec3(uv, l0)).rgb;
    vec3 b = texture(specEnv, vec3(uv, min(l0 + 1.0, float(specLayers - 1)))).rgb;
    return mix(a, b, l - l0);
}
vec2 envBRDF(float nv, float rough) {   // Karis' analytic fit of the split-sum BRDF table
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}
const vec2 poisson[16] = vec2[](vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725), vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760), vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464), vec2(-0.38277543, 0.27676845),
    vec2(0.97484398, 0.75648379), vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420), vec2(-0.26496911, -0.41893023),
    vec2(0.79197514, 0.19090188), vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590), vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790));
float sunShadow(vec3 p, vec3 n) {
    vec4 q = L * vec4(p + n * 0.6, 1.0);
    vec3 s = q.xyz / q.w * 0.5 + 0.5;
    if (s.x <= 0.0 || s.y <= 0.0 || s.x >= 1.0 || s.y >= 1.0 || s.z >= 1.0) return 1.0;
    float sum = 0.0;
    for (int i = 0; i < 16; i++) sum += texture(shadowTex, vec3(s.xy + poisson[i] * shadowTexel * 2.5, s.z - 0.0004));
    return sum / 16.0;
}
void main() {
    vec3 albedo = baseColor;
    if (hasBase == 1) albedo = texture(baseTex, v_uv).rgb * (1.0 + (brightness - 1.0) * v_jit);   // the brightness range scales the cloth texels
    albedo = srgb2lin(clamp(albedo, 0.0, 1.0));
    float rough = roughDefault, metal = metalDefault;
    if (hasMR == 1) { vec3 mr = texture(mrTex, v_uv).rgb; rough = mr.g; metal = mr.b; }
    rough = clamp(rough, 0.08, 1.0);
    vec3 N = normalize(v_n);
    vec3 T = normalize(v_t - N * dot(v_t, N));
    vec3 B = normalize(v_b - N * dot(v_b, N) - T * dot(v_b, T));
    if (!gl_FrontFacing) { N = -N; T = -T; B = -B; }
    if (hasNormal == 1) {
        vec3 tn = texture(normalTex, v_uv).rgb * 2.0 - 1.0;
        N = normalize(tn.x * T + tn.y * B + max(tn.z, 0.05) * N);
    }
    vec3 V = normalize(camPos - v_pw);
    if (dot(N, V) < 0.0) N = normalize(N - V * dot(N, V) * 1.01);      // bent normals facing away: keep them just facing the camera
    float nv = clamp(dot(N, V), 1e-3, 1.0);
    float ao = texelFetch(aoTex, ivec2(gl_FragCoord.xy), 0).r;
    vec3 F0 = mix(vec3(0.04), albedo, metal);
    // sun
    vec3 Ls = normalize(sunDir);
    float nl = clamp(dot(N, Ls), 0.0, 1.0);
    vec3 H = normalize(Ls + V);
    float nh = clamp(dot(N, H), 0.0, 1.0), vh = clamp(dot(V, H), 0.0, 1.0);
    float a = rough * rough, a2 = a * a;
    float D = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    float G = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - vh, 5.0);
    vec3 spec = D * G * F / max(4.0 * nv * nl, 1e-4);
    vec3 kd = (1.0 - F) * (1.0 - metal);
    float sh_ = sunShadow(v_pw, normalize(v_n));
    vec3 direct = (kd * albedo / PI + spec) * sunColor * nl * sh_ * PI;
    // environment
    vec3 diffIBL = irradiance(N) * albedo * (1.0 - metal) * PI;
    vec2 ab = envBRDF(nv, rough);
    vec3 R = reflect(-V, N);
    vec3 specIBL = envSpec(R, rough) * (F0 * ab.x + ab.y);
    float specAO = clamp(pow(nv + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    vec3 ambient = (diffIBL * ao + specIBL * specAO) * envScale;
    vec3 col = (direct + ambient) * exposure;
    o_col = vec4(lin2srgb(toneMap == 1 ? aces(col) : neutral(col)), 1.0);
    if (debugMode == 1) o_col = vec4(lin2srgb(albedo), 1.0);
    if (debugMode == 2) o_col = vec4(N * 0.5 + 0.5, 1.0);
    if (debugMode == 3) o_col = vec4(vec3(ao), 1.0);
    if (debugMode == 4) o_col = vec4(vec3(sh_), 1.0);
    if (debugMode == 5) o_col = vec4(lin2srgb(diffIBL * envScale), 1.0);
    if (debugMode == 6) o_col = vec4(lin2srgb(specIBL * envScale), 1.0);
    if (debugMode == 7) o_col = vec4(rough, metal, 0.0, 1.0);
}
"""
_GROUND_VS = """
#version 430
uniform mat3 R; uniform vec3 C; uniform vec4 proj; uniform vec2 size; uniform float near; uniform float far;
in vec3 in_pos;
out vec3 v_pw;
void main() {
    vec3 pc = transpose(R) * (in_pos - C);
    float d = pc.x;
    float x = (2.0 * proj.x / size.x) * pc.y + (2.0 * proj.z / size.x - 1.0) * d;
    float y = (2.0 * proj.y / size.y) * pc.z + (1.0 - 2.0 * proj.w / size.y) * d;
    float z = d * (far + near) / (far - near) - 2.0 * far * near / (far - near);
    gl_Position = vec4(x, y, z, d);
    v_pw = in_pos;
}
"""
_GROUND_FS = """
#version 430
uniform sampler2DShadow shadowTex; uniform mat4 L; uniform float shadowTexel;
in vec3 v_pw;
out vec4 o_col;
const vec2 poisson[16] = vec2[](vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725), vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760), vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464), vec2(-0.38277543, 0.27676845),
    vec2(0.97484398, 0.75648379), vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420), vec2(-0.26496911, -0.41893023),
    vec2(0.79197514, 0.19090188), vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590), vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790));
void main() {
    vec4 q = L * vec4(v_pw + vec3(0.0, 0.0, 0.5), 1.0);
    vec3 s = q.xyz / q.w * 0.5 + 0.5;
    float lit = 1.0;
    if (s.x > 0.0 && s.y > 0.0 && s.x < 1.0 && s.y < 1.0 && s.z < 1.0) {
        float sum = 0.0;
        for (int i = 0; i < 16; i++) sum += texture(shadowTex, vec3(s.xy + poisson[i] * shadowTexel * 6.0, s.z - 0.001));
        lit = sum / 16.0;
    }
    o_col = vec4(lit, 0.0, 0.0, 1.0);
}
"""


# ----------------------------------------------------------------------------- environment / IBL
def _dirs(w: int, h: int) -> np.ndarray:
    """(h, w, 3) unit directions of an equirect (z up; u = azimuth from -pi, v = polar angle from the zenith), texel centres."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    phi = (u - 0.5) * 2 * np.pi
    th = v * np.pi
    P, T = np.meshgrid(phi, th)
    return np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)


def environment(sky_rgb=(0.62, 0.70, 0.82), ground_rgb=(0.32, 0.30, 0.28), sun_dir=(0.6, 0.3, 0.74)) -> np.ndarray:
    """Sky/ground environment, (ENV_H, ENV_W, 3) linear radiance (relative units, ~1 = the sky): a bright horizon band, a sky that is
    brighter towards the sun, a darker ground of `ground_rgb` (the bounce from the street) and soft daylight panels like the light
    tents of a model viewer (front-top and back-top), so faces and folds are modelled even under an overcast sky."""
    d = _dirs(ENV_W, ENV_H)
    z = d[..., 2]
    sky = np.asarray(sky_rgb, float)
    ground = np.asarray(ground_rgb, float)
    hor = 0.5 * (sky + 1.0)
    up = np.clip(z, 0, 1)[..., None]
    env = np.where(z[..., None] >= 0, hor * (1 - up) + sky * up, hor * 0.6 * np.exp(z[..., None] * 6) + ground * (1 - np.exp(z[..., None] * 6)))
    s = np.asarray(sun_dir, float)
    s = s / np.linalg.norm(s)
    cos_s = np.clip((d * s).sum(-1), -1, 1)
    env = env * (1.0 + 0.8 * np.clip(cos_s, 0, 1) ** 4 * (z >= -0.05))[..., None]   # circumsolar glow
    for az, el, size, gain in ((np.arctan2(s[1], s[0]), 35.0, 32.0, 1.4), (np.arctan2(s[1], s[0]) + np.pi, 30.0, 28.0, 0.9)):
        c = np.array([np.cos(np.radians(el)) * np.cos(az), np.cos(np.radians(el)) * np.sin(az), np.sin(np.radians(el))])
        ang = np.degrees(np.arccos(np.clip((d * c).sum(-1), -1, 1)))
        env = env + gain * np.clip(1.0 - ang / size, 0, 1)[..., None] ** 2 * 0.5 * (1.0 + sky)
    return env.astype(np.float32)


def _sh9(env: np.ndarray) -> np.ndarray:
    """9 spherical-harmonic coefficients (9, 3) of an equirect radiance map."""
    h, w = env.shape[:2]
    d = _dirs(w, h)
    th = (np.arange(h) + 0.5) / h * np.pi
    dw = (2 * np.pi / w) * (np.pi / h) * np.sin(th)[:, None]
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    basis = [0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
             0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)]
    # GLSL order used by `irradiance`: L00, L1-1 (y), L10 (z), L11 (x), L2-2 (xy), L2-1 (yz), L20, L21 (xz), L22
    return np.stack([(env * (b * dw)[..., None]).sum(axis=(0, 1)) for b in basis]).astype(np.float32)


def _prefilter(env: np.ndarray) -> np.ndarray:
    """(len(SPEC_ROUGHNESS), ENV_H, ENV_W, 3): the environment convolved with the GGX lobe of each roughness (normal = view = reflection)."""
    sw, sh = ENV_W, ENV_H
    small = env
    dsrc = _dirs(sw, sh).reshape(-1, 3)
    th = (np.arange(sh) + 0.5) / sh * np.pi
    dw = ((2 * np.pi / sw) * (np.pi / sh) * np.sin(th)[:, None] * np.ones((1, sw))).reshape(-1)
    rad = small.reshape(-1, 3)
    dout = _dirs(ENV_W, ENV_H).reshape(-1, 3)
    out = []
    for r in SPEC_ROUGHNESS:
        if r == 0.0:
            out.append(env)
            continue
        a2 = max(r * r, 1e-3) ** 2
        res = np.zeros((len(dout), 3))
        for i in range(0, len(dout), 1024):
            c = np.clip(dout[i:i + 1024] @ dsrc.T, 0, 1)                    # n.l with n = r
            D = a2 / (np.pi * (c * c * (a2 - 1) + 1) ** 2)
            wgt = D * c * dw[None, :]
            res[i:i + 1024] = (wgt @ rad) / np.maximum(wgt.sum(axis=1, keepdims=True), 1e-12)
        out.append(res.reshape(ENV_H, ENV_W, 3).astype(np.float32))
    return np.stack(out).astype(np.float32)


# ----------------------------------------------------------------------------- the GL worker
class _GL:
    """One OpenGL context, owned by one thread (WGL/GLX contexts are bound to the thread that made them current): every render is a
    job run on that thread, so Studio's request threads and the generator can all call `render`."""
    _inst: "_GL | None" = None
    _inst_lock = threading.Lock()

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._loop, name="tosdg-gl", daemon=True)
        self.thread.start()
        self.ready.wait()
        if self.error:
            raise self.error

    @classmethod
    def get(cls) -> "_GL":
        with cls._inst_lock:
            if cls._inst is None:
                cls._inst = _GL()
            return cls._inst

    def _loop(self):
        try:
            import moderngl
            self.ctx = moderngl.create_standalone_context(require=430)
            self.name = self.ctx.info["GL_RENDERER"]
            self._build()
        except Exception as e:  # noqa: BLE001 - reported to the caller
            self.error = RuntimeError(f"no OpenGL 4.3 GPU context ({e}); use the CPU mesh renderer")
            self.ready.set()
            return
        self.ready.set()
        while True:
            fn, box, done = self.q.get()
            try:
                box["result"] = fn(self)
            except Exception as e:  # noqa: BLE001 - re-raised in the caller's thread
                box["error"] = e
            done.set()

    def run(self, fn):
        box, done = {}, threading.Event()
        self.q.put((fn, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["result"]

    # -- resources
    def _build(self):
        c = self.ctx
        self.p_geo = c.program(vertex_shader=_GEO_VS, fragment_shader=_GEO_FS)
        self.p_shadow = c.program(vertex_shader=_SHADOW_VS, fragment_shader=_SHADOW_FS)
        self.p_ssao = c.program(vertex_shader=_QUAD_VS, fragment_shader=_SSAO_FS)
        self.p_blur = c.program(vertex_shader=_QUAD_VS, fragment_shader=_BLUR_FS)
        self.p_shade = c.program(vertex_shader=_SHADE_VS, fragment_shader=_SHADE_FS)
        self.p_ground = c.program(vertex_shader=_GROUND_VS, fragment_shader=_GROUND_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.quad_vbo = c.buffer(quad.tobytes())
        self.vao_ssao = c.vertex_array(self.p_ssao, [(self.quad_vbo, "2f", "in_xy")])
        self.vao_blur = c.vertex_array(self.p_blur, [(self.quad_vbo, "2f", "in_xy")])
        self.shadow_tex = c.depth_texture((SHADOW_SIZE, SHADOW_SIZE))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.filter = (0x2601, 0x2601)        # GL_LINEAR: hardware 2x2 PCF
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = c.framebuffer(depth_attachment=self.shadow_tex)
        rng = np.random.default_rng(3)
        k = []
        while len(k) < 32:
            v = rng.uniform(-1, 1, 3)
            v[2] = abs(v[2])
            if 0.1 < np.linalg.norm(v) <= 1.0:
                v = v / np.linalg.norm(v) * (0.15 + 0.85 * (len(k) / 32) ** 2)
                v[2] = max(v[2], 0.15)
                k.append(v)
        self.kernel = np.array(k, np.float32)
        self.textures: dict = {}                # id(array) -> (array, texture); the array is kept so its id stays unique
        self.env_cache: dict = {}
        self.size_cache: dict = {}

    def texture(self, arr: np.ndarray, srgb_data: bool):
        key = id(arr)
        hit = self.textures.get(key)
        if hit is not None and hit[0] is arr:
            return hit[1]
        if len(self.textures) > 24:
            for _, (_, t) in list(self.textures.items()):
                t.release()
            self.textures.clear()
        a = np.ascontiguousarray(np.asarray(arr, np.uint8))                 # row 0 = t 0 = glTF v 0 (the top of the image): no flip
        tex = self.ctx.texture((a.shape[1], a.shape[0]), 3, a.tobytes(), alignment=1)
        tex.build_mipmaps()
        tex.anisotropy = 16.0
        tex.repeat_x = tex.repeat_y = False
        self.textures[key] = (arr, tex)
        return tex

    def env(self, sky: tuple, ground: tuple, sun: tuple):
        key = tuple(np.round(np.r_[sky, ground, sun], 3))
        if key not in self.env_cache:
            if len(self.env_cache) > 16:
                for _, (_, t) in self.env_cache.items():
                    t.release()
                self.env_cache.clear()
            e = environment(sky, ground, sun)
            pre = _prefilter(e)
            data = np.ascontiguousarray(pre).astype(np.float32)            # row 0 = t 0 = the zenith, as envSpec samples it
            tex = self.ctx.texture_array((ENV_W, ENV_H, len(SPEC_ROUGHNESS)), 3, data.tobytes(), dtype="f4")
            tex.filter = (0x2601, 0x2601)
            tex.repeat_y = False
            self.env_cache[key] = (_sh9(e), tex)
        return self.env_cache[key]

    def targets(self, W: int, H: int):
        key = (W, H)
        if key not in self.size_cache:
            for _, d in list(self.size_cache.items()):
                for r in d.values():
                    r.release()
            self.size_cache.clear()
            c = self.ctx
            t = {"pos": c.texture((W, H), 4, dtype="f4"), "nrm": c.texture((W, H), 4, dtype="f4"), "geo_depth": c.depth_renderbuffer((W, H)),
                 "ao": c.texture((W, H), 1, dtype="f4"), "ao2": c.texture((W, H), 1, dtype="f4"),
                 "col_ms": c.renderbuffer((W, H), 4, samples=MSAA, dtype="f4"), "dep_ms": c.depth_renderbuffer((W, H), samples=MSAA),
                 "col": c.texture((W, H), 4, dtype="f4"), "ground": c.texture((W, H), 4, dtype="f4"),
                 "ground_depth": c.depth_renderbuffer((W, H))}
            for k in ("pos", "nrm", "ao", "ao2"):
                t[k].filter = (0x2600, 0x2600)
            t["fbo_geo"] = c.framebuffer([t["pos"], t["nrm"]], t["geo_depth"])
            t["fbo_ao"] = c.framebuffer([t["ao"]])
            t["fbo_ao2"] = c.framebuffer([t["ao2"]])
            t["fbo_ms"] = c.framebuffer([t["col_ms"]], t["dep_ms"])
            t["fbo_col"] = c.framebuffer([t["col"]])
            t["fbo_ground"] = c.framebuffer([t["ground"]], t["ground_depth"])
            self.size_cache[key] = t
        return self.size_cache[key]


def available() -> tuple[bool, str]:
    """(True, GL renderer name) when the GPU renderer can run here, else (False, reason)."""
    try:
        return True, _GL.get().name
    except Exception as e:  # noqa: BLE001
        return False, str(e)


# ----------------------------------------------------------------------------- one frame
def _set(prog, name, value):
    if name in prog:
        prog[name].value = value


def _camera_uniforms(prog, cam: Camera, S: int, W: int, H: int):
    _set(prog, "R", tuple(np.asarray(cam.R, float).T.reshape(-1)))        # GLSL mat3 is column-major: R^T row-major = R column-major
    _set(prog, "C", tuple(float(x) for x in cam.position))
    _set(prog, "proj", (cam.fx * S, cam.fy * S, cam.cx * S, cam.cy * S))
    _set(prog, "size", (float(W), float(H)))
    _set(prog, "near", 1.0)
    _set(prog, "far", 100000.0)


def _light_matrix(pos: np.ndarray, sun: np.ndarray) -> tuple[np.ndarray, float]:
    """Orthographic light matrix (column-major flattening for GL) looking along -sun, fitted to the mesh; and the texel size in uv."""
    c = 0.5 * (pos.min(axis=0) + pos.max(axis=0))
    r = float(np.linalg.norm(pos - c, axis=1).max()) + 5.0
    f = -sun / np.linalg.norm(sun)
    up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, f)
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = x / r, y / r, f / r
    M[:3, 3] = -M[:3, :3] @ c
    return M, 1.0 / SHADOW_SIZE


def _frame(gl: _GL, mesh: UniformMesh, fk_world: FKResult, camera: Camera, colors: dict, lighting: dict, brightness: float,
           backplate: np.ndarray | None, ground: bool = True) -> dict:
    ctx = gl.ctx
    S = SUPERSAMPLE if max(camera.width, camera.height) * SUPERSAMPLE <= MAX_CANVAS_SIDE else 1
    W, H = camera.width * S, camera.height * S
    t = gl.targets(W, H)

    pos, nrm = skin_mesh(mesh, fk_world)
    if mesh.uv is not None:
        Tb, Bb = tangent_frame(mesh)
        tan, bit = _blend_dirs(mesh, fk_world, Tb), _blend_dirs(mesh, fk_world, Bb)
        uv = np.asarray(mesh.uv, np.float32)
    else:
        tan = np.tile([1.0, 0.0, 0.0], (mesh.n_vertices, 1))
        bit = np.tile([0.0, 1.0, 0.0], (mesh.n_vertices, 1))
        uv = np.zeros((mesh.n_vertices, 2), np.float32)
    jit = mesh.vertex_jitter if mesh.vertex_jitter is not None else np.zeros(mesh.n_vertices)
    vdata = np.hstack([pos, nrm, tan, bit, uv, np.asarray(jit, float)[:, None]]).astype(np.float32)
    vbo = ctx.buffer(vdata.tobytes())
    n_off = mesh.instance_tris if mesh.instance_tris is not None else len(mesh.triangles)
    tri_ids = np.arange(len(mesh.triangles))
    groups = []                                                            # (material, is officer, index buffer, count)
    for mi in range(len(mesh.materials)):
        for officer in (True, False):
            sel = (mesh.tri_material == mi) & ((tri_ids < n_off) == officer)
            if sel.any():
                idx = mesh.triangles[sel].astype(np.int32)
                groups.append((mi, officer, ctx.buffer(idx.tobytes()), int(idx.size)))
    fmt = "3f 3f 3f 3f 2f 1f"
    names = ["in_pos", "in_nrm", "in_tan", "in_bit", "in_uv", "in_jit"]
    released = [vbo] + [g[2] for g in groups]
    try:
        # 1. sun shadow map
        sun = np.asarray(lighting["sun_direction"], float)
        sun = sun / np.linalg.norm(sun)
        Lm, texel = _light_matrix(pos, sun)
        Lflat = tuple(Lm.T.reshape(-1))
        gl.shadow_fbo.use()
        gl.shadow_fbo.clear(depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.disable(ctx.CULL_FACE)
        ctx.front_face = "cw"          # the world frame is left-handed (UE): outward triangles wind clockwise on screen
        ctx.polygon_offset = (2.0, 4.0)
        gl.p_shadow["L"].value = Lflat
        for _, _, ib, _ in groups:
            vao = ctx.vertex_array(gl.p_shadow, [(vbo, "3f 48x", "in_pos")], ib)
            vao.render()
            vao.release()
        ctx.polygon_offset = (0.0, 0.0)

        # 2. geometry pre-pass: camera-frame position, normal, instance id
        t["fbo_geo"].use()
        t["fbo_geo"].clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        _camera_uniforms(gl.p_geo, camera, S, W, H)
        for _, officer, ib, _ in groups:
            gl.p_geo["inst"].value = 1.0 if officer else 2.0
            vao = ctx.vertex_array(gl.p_geo, [(vbo, "3f 3f 36x", "in_pos", "in_nrm")], ib)
            vao.render()
            vao.release()
        geo = np.frombuffer(t["fbo_geo"].read(components=4, dtype="f4", attachment=0), np.float32).reshape(H, W, 4)[::-1]

        # 3. ambient occlusion
        ctx.disable(ctx.DEPTH_TEST)
        t["fbo_ao"].use()
        t["pos"].use(0)
        t["nrm"].use(1)
        gl.p_ssao["posTex"].value, gl.p_ssao["nrmTex"].value = 0, 1
        gl.p_ssao["proj"].value = (camera.fx * S, camera.fy * S, camera.cx * S, camera.cy * S)
        gl.p_ssao["size"].value = (float(W), float(H))
        gl.p_ssao["radius"].value = 9.0
        gl.p_ssao["nk"].value = 32
        gl.p_ssao["kernel"].write(gl.kernel.tobytes())
        gl.vao_ssao.render(mode=ctx.TRIANGLE_STRIP)
        t["fbo_ao2"].use()
        t["ao"].use(0)
        t["pos"].use(1)
        gl.p_blur["aoTex"].value, gl.p_blur["posTex"].value = 0, 1
        gl.vao_blur.render(mode=ctx.TRIANGLE_STRIP)

        # 4. shading
        top = backplate[: max(1, backplate.shape[0] // 3)].reshape(-1, 3).mean(axis=0) / 255.0 if backplate is not None else np.array([0.62, 0.7, 0.82])
        low = backplate[-max(1, backplate.shape[0] // 3):].reshape(-1, 3).mean(axis=0) / 255.0 if backplate is not None else np.array([0.32, 0.3, 0.28])
        sky = tuple(np.clip(0.55 * np.array(SKY_BASE) + 0.45 * top ** 2.2 * 1.6, 0.05, 2.0))   # a neutral daylight sky, tinted by the photo
        ground = tuple(np.clip(0.22 + 0.6 * low ** 2.2, 0.05, 1.0))          # sunlit street bounce: never black under the arms
        shc, spec_tex = gl.env(sky, ground, tuple(sun))
        p = gl.p_shade
        _camera_uniforms(p, camera, S, W, H)
        p["camPos"].value = tuple(float(x) for x in camera.position)
        p["sunDir"].value = tuple(sun)
        p["sunColor"].value = tuple(np.asarray(lighting["sun_color_rgb"], float) * lighting["sun_intensity"] * SUN_GAIN)
        p["sh"].write(shc.tobytes())
        p["envScale"].value = float(lighting["ambient_intensity"]) * ENV_GAIN
        p["specLayers"].value = len(SPEC_ROUGHNESS)
        p["exposure"].value = float(2.0 ** lighting.get("exposure_ev", 0.0))
        p["brightness"].value = float(brightness)
        p["toneMap"].value = 1 if TONE_MAP == "aces" else 0
        if "debugMode" in p:
            p["debugMode"].value = DEBUG
        p["L"].value = Lflat
        p["shadowTexel"].value = texel
        for name, unit in (("baseTex", 0), ("normalTex", 1), ("mrTex", 2), ("aoTex", 3), ("shadowTex", 4), ("specEnv", 5)):
            p[name].value = unit
        t["ao2"].use(3)
        gl.shadow_tex.use(4)
        spec_tex.use(5)
        t["fbo_ms"].use()
        t["fbo_ms"].clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        base = material_albedo(mesh, colors, brightness)                  # untextured cloth: the brightness range is in the albedo
        mat_tex = _tex_of(mesh)
        for mi, _, ib, _ in groups:
            m = mesh.materials[mi]
            has_b = int(mat_tex[mi] >= 0)
            bound = [(gl.texture(mesh.textures[m[key]], key == "texture"), unit) for key, unit in (("texture", 0), ("normal_texture", 1), ("mr_texture", 2))
                     if m.get(key) is not None]                          # create/upload every texture first: creating one rebinds a unit
            for tex, unit in bound:
                tex.use(unit)
            t["ao2"].use(3)
            gl.shadow_tex.use(4)
            spec_tex.use(5)
            for key, flag in (("normal_texture", "hasNormal"), ("mr_texture", "hasMR")):
                p[flag].value = int(m.get(key) is not None)
            p["hasBase"].value = has_b
            p["baseColor"].value = tuple(float(x) for x in base[mi])
            p["roughDefault"].value = 0.75 if m.get("jitter") else 0.45
            p["metalDefault"].value = 0.0
            vao = ctx.vertex_array(p, [(vbo, fmt, *names)], ib)
            vao.render()
            vao.release()
        ctx.copy_framebuffer(t["fbo_col"], t["fbo_ms"])
        col = np.frombuffer(t["fbo_col"].read(components=4, dtype="f4"), np.float32).reshape(H, W, 4)[::-1]

        # 5. ground shadow
        ground_lit = None
        if ground and GROUND_SHADOW > 0 and backplate is not None:
            c0 = 0.5 * (pos.min(axis=0) + pos.max(axis=0))
            r = 2500.0
            quad = np.array([[c0[0] - r, c0[1] - r, 0], [c0[0] + r, c0[1] - r, 0], [c0[0] - r, c0[1] + r, 0], [c0[0] + r, c0[1] + r, 0]], np.float32)
            qb = ctx.buffer(quad.tobytes())
            released.append(qb)
            pg = gl.p_ground
            _camera_uniforms(pg, camera, S, W, H)
            pg["L"].value = Lflat
            pg["shadowTexel"].value = texel
            pg["shadowTex"].value = 4
            gl.shadow_tex.use(4)
            t["fbo_ground"].use()
            t["fbo_ground"].clear(1.0, 0.0, 0.0, 0.0, depth=1.0)
            ctx.disable(ctx.DEPTH_TEST)
            vao = ctx.vertex_array(pg, [(qb, "3f", "in_pos")])
            vao.render(mode=ctx.TRIANGLE_STRIP)
            vao.release()
            ground_lit = np.frombuffer(t["fbo_ground"].read(components=4, dtype="f4"), np.float32).reshape(H, W, 4)[::-1, :, 0]
    finally:
        for b in released:
            b.release()
    return {"S": S, "geo": geo, "col": col, "ground": ground_lit, "sun_up": float(max(sun[2], 0.0))}


def _down(a: np.ndarray, S: int) -> np.ndarray:
    H, W = a.shape[0] // S, a.shape[1] // S
    return a.reshape(H, S, W, S, *a.shape[2:]).mean(axis=(1, 3))


def render(mesh: UniformMesh, fk_world: FKResult, camera: Camera, backplate: Image.Image, colors: dict[str, list[int]],
           lighting: dict, brightness: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb uint8 HxWx3, mask bool HxW) like render_mesh.render; the mask is the officer's own coverage (>= 0.5 of the samples)."""
    gl = _GL.get()
    W, H = camera.width, camera.height
    plate = np.asarray(backplate.convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.float32)
    f = gl.run(lambda g: _frame(g, mesh, fk_world, camera, colors, lighting, brightness, plate.astype(np.uint8)))
    S = f["S"]
    col = _down(f["col"], S)
    rgb_p, alpha = col[..., :3], col[..., 3:4]
    img = plate / 255.0
    if f["ground"] is not None:
        lit = _down(f["ground"], S)[..., None]
        k = GROUND_SHADOW * (0.35 + 0.65 * min(1.0, lighting["sun_intensity"])) * (0.5 + 0.5 * f["sun_up"])
        img = img * (1.0 - k * (1.0 - lit))
    out = rgb_p + (1.0 - alpha) * img
    mask = _down((f["geo"][..., 3] == 1.0).astype(np.float32), S) >= 0.5
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), mask


def render_layer(mesh: UniformMesh, fk_world: FKResult, camera: Camera, colors: dict[str, list[int]], lighting: dict,
                 brightness: float = 1.0, backplate: Image.Image | None = None) -> np.ndarray:
    """The officer alone as an RGBA uint8 image (alpha = anti-aliased coverage), like render_mesh.render_layer. `backplate` (optional)
    only tints the sky light, as in `render`; no ground shadow."""
    gl = _GL.get()
    plate = None if backplate is None else np.asarray(backplate.convert("RGB").resize((camera.width, camera.height), Image.BILINEAR), np.uint8)
    f = gl.run(lambda g: _frame(g, mesh, fk_world, camera, colors, lighting, brightness, plate, ground=False))
    col = _down(f["col"], f["S"])
    a = col[..., 3:4]
    rgb = col[..., :3] / np.maximum(a, 1e-6)
    out = np.zeros((camera.height, camera.width, 4), np.uint8)
    out[..., :3] = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    out[..., 3] = (np.clip(a[..., 0], 0, 1) * 255 + 0.5).astype(np.uint8)
    return out
