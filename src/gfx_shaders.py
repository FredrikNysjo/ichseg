raycast_vs = """
#version 410

uniform int u_projection_mode=0;
uniform mat4 u_mv;
uniform mat4 u_mvp;

out vec3 v_ray_origin;
out vec3 v_ray_dir;

const int CUBE_INDICES[] = int[](3, 2, 7, 6, 4, 2, 0, 3, 1, 7, 5, 4, 1, 0);

void main()
{
    int idx = CUBE_INDICES[gl_VertexID];
    v_ray_origin = vec3(idx % 2, (idx / 2) % 2, idx / 4) - 0.5;
    v_ray_dir = v_ray_origin - vec3(inverse(u_mv) * vec4(0.0, 0.0, 0.0, 1.0));
    if (bool(u_projection_mode))
        v_ray_dir = -vec3(inverse(u_mv)[2]);  // For orthographic projection
    gl_Position = u_mvp * vec4(v_ray_origin, 1.0);
}
"""

raycast_fs = """
#version 410

#define MAX_STEPS 1000

uniform int u_label=0;
uniform int u_show_mask=1;
uniform int u_show_mpr=1;
uniform vec3 u_mpr_planes=vec3(0.0);
uniform vec2 u_level_range=vec2(0.0, 1.0);
uniform vec3 u_extent=vec3(1.0);
uniform vec4 u_brush;
uniform mat4 u_mvp;
uniform sampler3D u_volume;
uniform sampler3D u_mask;

in vec3 v_ray_origin;
in vec3 v_ray_dir;
out vec4 rt_color;

bool intersectBox(vec3 ray_origin, vec3 ray_dir_inv, vec3 aabb[2], out float tmin, out float tmax)
{
    vec3 t1 = (aabb[0] - ray_origin) * ray_dir_inv;
    vec3 t2 = (aabb[1] - ray_origin) * ray_dir_inv;
    tmin = max(min(t1[0], t2[0]), max(min(t1[1], t2[1]), min(t1[2], t2[2])));
    tmax = min(max(t1[0], t2[0]), min(max(t1[1], t2[1]), max(t1[2], t2[2])));
    return (tmax - tmin) > 0.0;
}

vec3 hsv2rgb(float h, float s, float v)
{
    vec3 k = fract(vec3(5.0, 3.0, 1.0) / 6.0 + h) * 6.0;
    return v - v * s * clamp(min(k, 4.0 - k), 0.0, 1.0);
}

void main()
{
    ivec3 res = textureSize(u_volume, 0).xyz;

    vec3 ray_origin = v_ray_origin;
    vec3 ray_dir = normalize(v_ray_dir);
    vec3 ray_dir_inv = clamp(1.0 / ray_dir, -9999.0, 9999.0);
    vec3 ray_step = 2.0 * ray_dir / res.xxx;  // FIXME
    float jitter = fract(dot(vec2(0.754877, 0.569840), gl_FragCoord.xy));
    float tmin, tmax;
    vec3 aabb[] = vec3[](vec3(-0.5), vec3(0.5));
    intersectBox(ray_origin, ray_dir_inv, aabb, tmin, tmax);

    if (bool(u_show_mpr)) {
        tmin = tmax; jitter = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (abs(ray_dir[i]) < 0.001)
                continue;  // Avoid grazing intersections with plane
            float tmin_plane, tmax_plane;
            vec3 aabb_plane[] = vec3[](aabb[0], aabb[1]);
            aabb_plane[0][i] = max(-0.5, min(0.5, u_mpr_planes[i] - 1e-4));
            aabb_plane[1][i] = max(-0.5, min(0.5, u_mpr_planes[i] + 1e-4));
            if (intersectBox(ray_origin, ray_dir_inv, aabb_plane, tmin_plane, tmax_plane) && tmin_plane < tmin) {
                tmin = tmin_plane;
                tmax = tmax_plane;
            }
        }
        ray_origin = ray_origin + tmin * ray_dir;
        if (tmax <= tmin) discard;
    }

    // Do ray marching (in both volumes)
    vec3 p = ray_origin;
    vec2 intensity = vec2(-999999.0);
    int nsteps = int(ceil((tmax - tmin) / length(ray_step)));
    for (int i = 0; i < min(MAX_STEPS, nsteps); ++i) {
        p = ray_origin + ray_step * (i + jitter);
        intensity[0] = max(intensity[0], texture(u_volume, p + 0.5).r);
        intensity[1] = max(intensity[1], texture(u_mask, p + 0.5).r);
    }
    intensity[0] = max(0.0, intensity[0] - u_level_range[0]) / (u_level_range[1] - u_level_range[0]);

    // Compute image gradient for mask volume
    vec2 mask_grad = vec2(0.0);
    mask_grad.x += step(0.5, texture(u_mask, p + dFdx(p) + 0.5).r);
    mask_grad.x -= step(0.5, texture(u_mask, p - dFdx(p) + 0.5).r);
    mask_grad.y += step(0.5, texture(u_mask, p + dFdy(p) + 0.5).r);
    mask_grad.y -= step(0.5, texture(u_mask, p - dFdy(p) + 0.5).r);

    vec4 output_color = vec4(intensity[0]);

    // Draw segmentation mask
    vec3 label_color = hsv2rgb(fract(u_label * 0.618034), 0.5, 1.0);
    if (bool(u_show_mask) && !bool(u_show_mpr)) {
        output_color.rgb = max(output_color.rgb, label_color * intensity[1]);
    }
    if (bool(u_show_mask) && bool(u_show_mpr)) {
        float outline = clamp(length(mask_grad), 0.0, 1.0);
        outline *= 1.0 - step(0.05, max(length(dFdx(p)), length(dFdy(p))));
        output_color.rgb = mix(output_color.rgb, label_color, mix(intensity[1], outline, 0.7));
    }

    // Draw brush shape (as white highlight)
    if (length((u_brush.xyz - p) * u_extent / u_extent.x) < u_brush.w / res.x) {
        output_color.rgb += vec3(0.15);
    }

    // Output fragment color and depth (the latter for picking)
    rt_color = output_color;
    vec4 clip_pos = u_mvp * vec4(p, 1.0);
    gl_FragDepth = clamp((clip_pos.z / clip_pos.w) * 0.5 + 0.5, 0.0, 1.0);
}
"""

polygon_vs = """
#version 410

uniform mat4 u_mvp;

layout(location = 0) in vec4 a_position;

void main()
{
    gl_Position = u_mvp * a_position;
}
"""

polygon_fs = """
#version 410

out vec4 rt_color;

void main()
{
    rt_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

background_vs = """
#version 410

out vec2 v_texcoord;

void main()
{
    v_texcoord = vec2(gl_VertexID % 2, gl_VertexID / 2);
    gl_Position = vec4(v_texcoord * 2.0 - 1.0, 0.0, 1.0);
}
"""

background_fs = """
#version 410

uniform vec3 u_bg_color1;
uniform vec3 u_bg_color2;

in vec2 v_texcoord;
out vec4 rt_color;

void main()
{
    rt_color = vec4(mix(u_bg_color2, u_bg_color1, v_texcoord.y), 0.0);
}
"""
