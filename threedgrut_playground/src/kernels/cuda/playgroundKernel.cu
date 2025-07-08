// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#define __PLAYGROUND__MODE__ 1

#include <playground/pipelineParameters.h>

extern "C"
{
    #ifndef __PLAYGROUND__PARAMS__
    __constant__ PlaygroundPipelineParameters params;
    #define __PLAYGROUND__PARAMS__ 1
    #endif
}

#include <optix.h>
#include <playground/kernels/cuda/trace.cuh>
#include <playground/kernels/cuda/materials.cuh>

constexpr uint32_t MAX_BOUNCES = 32;           // Maximum number of mirror material bounces only (irrelevant to pbr)
constexpr uint32_t TIMEOUT_ITERATIONS = 1000;  // Terminate ray after max iterations to avoid infinite loop
constexpr float REFRACTION_EPS_SHIFT = 1e-5;   // Add eps amount to refracted rays pos to avoid repeated collisions


extern "C" __global__ void __raygen__rg() {

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    float3 rayOrigin    = params.rayWorldOrigin(idx);
    float3 rayDirection = params.rayWorldDirection(idx);
    // Ray coordinates in pixels
    const int rx = fminf(idx.x, params.frameBounds.x);
    const int ry = fminf(idx.y, params.frameBounds.y);

    // Initialize Payload
    HybridRayPayload payload;
    payload.t_hit = 0.0;
    payload.rayOri = make_float3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
    payload.rayDir = make_float3(rayDirection.x, rayDirection.y, rayDirection.z);
    payload.numBounces = 0;
    payload.rndSeed = tea<16>(dim.x * idx.y + idx.x, params.frameNumber);
    const float ray_t_max = params.rayMaxT[idx.z][ry][rx];
    // Initialize Payload PBR fields
    payload.accumulatedColor = make_float3(0.0);
    payload.accumulatedAlpha = 0.0;
    payload.directLight = make_float3(0.0);
    payload.pathThroughput = make_float3(1.0);
    payload.bsdfValue = make_float3(1.0);
    payload.pbrNumBounces = 0;
    payload.rayMissed = false;

    // Initialize 3drt output buffers, we'll soon aggregate in them
    RayData rayData;
    rayData.initialize();
    payload.rayData = &rayData;

    float3 lastRayOri;          // Last ray origin used to trace Gaussians
    float3 lastRayDir;          // Last ray direction used to trace Gaussians
    unsigned int timeout = 0;

    // Loop always runs at least once.
    // Termination criteria:
    // 1. Ray missed surface (ray dir is 0), or
    // 2. PBR Materials: No remaining bounces, or
    // 3. PBR Materials: pathThroughput is zero -> no more color is reflected back, or
    // 4. Mirrors: No remaining bounces
    while (!payload.rayMissed &&
           (length(payload.pathThroughput) > 0.0001) &&
           payload.accumulatedAlpha < 0.995 &&
           (payload.pbrNumBounces < params.maxPBRBounces) &&
           (payload.numBounces < MAX_BOUNCES) &&
           (getNextTraceState() != PGRNDTraceTerminate))
    {
        // Fetch ray orig + dir to use for next mesh + Gaussian passes.
        const float3 rayOri = payload.rayOri;
        const float3 rayDir = payload.rayDir;

        // First trace the closest hit point with a mesh, and compute new ray direction + how much light scattered
        payload.bsdfValue = make_float3(1.0f);    // Will contain how much light is scattered along this path
        payload.nextEmissive = make_float3(0.0f); // Will contain how much additional light is emitted along this path
        // Trace the ray against BVH of reflective / refractive faces
        traceMesh(rayOri, rayDir, &payload);

        // Then perform volumetric radiance integration from ray origin to closest hit point (or infinity is ray missed)
        float next_ray_t = payload.rayMissed ? ray_t_max : payload.t_hit;
        float4 volumetricRadDns = traceGaussians(rayData, rayOri, rayDir, 1e-9, next_ray_t, &payload);
        float3 radiance = make_float3(volumetricRadDns.x, volumetricRadDns.y, volumetricRadDns.z);
        float density = volumetricRadDns.w;

        // -- Now accumulate the radiance collected along this path:
        // TODO (operel): The following will not suffice for pbr primitives with transmittance,
        // maybe bookkeep directLight and pathThroughput at mesh intersection points and backtrack?

        // Aggregate volumetric radiance
        payload.accumulatedAlpha += density * (1.0f - payload.accumulatedAlpha);
        payload.accumulatedColor += payload.pathThroughput * radiance; // Contribution of Gaussian radiance to path

        // Update attenuation of light according to volumetric integration:
        // This is by how much the Gaussians affect shading of PBR materials
        payload.directLight += radiance;                             // Gaussians also act as light sources
        payload.pathThroughput *= (1.0f - density);                  // Modulate throughput by volumetric transmittance

        // Update attenuation of light according to BRDF + BTDF
        // This is how much light was emitted by the surface + scattered along this path, modulated by the throughput
        payload.accumulatedColor += payload.pathThroughput * (payload.directLight + payload.nextEmissive);
        payload.pathThroughput *= payload.bsdfValue;

        // Keep this in case additional background sampling is done external to the shader
        lastRayOri = rayOri;
        lastRayDir = rayDir;

        timeout += 1;
        if (timeout > TIMEOUT_ITERATIONS)
            break;
    }

    const float3 background = getBackgroundColor(lastRayDir);
    payload.directLight += background;
    payload.pathThroughput *= (1.0f - payload.accumulatedAlpha);
    payload.accumulatedColor += payload.pathThroughput * payload.directLight;
    payload.accumulatedAlpha = clamp(payload.accumulatedAlpha, 0.0f, 1.0f);

    // Write back to global mem in launch params
    const float4 rgba = make_float4(payload.accumulatedColor.x, payload.accumulatedColor.y, payload.accumulatedColor.z,
                                    payload.accumulatedAlpha);

    writeRadianceDensityToOutputBuffer(rgba);
    writeUpdatedRaysToBuffer(lastRayOri, lastRayDir);
}

static __device__ __inline__ bool refract(float3& out_dir, const float3 ray_d, float3 normal,
                                          const float etai_over_etat, unsigned int& rndSeed)
{
    // Algorithm based on: https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/refraction
    float ri;
    if (dot(ray_d, normal) < 0.0)   // front_face ? (1.0/refraction_index)
    {
        ri = 1.0 / etai_over_etat;
    }
    else                            // back face? (refraction_index)
    {
        ri = etai_over_etat;
        normal = -normal;
    }

    // Move above normal update?
    float cos_theta = fminf(dot(-ray_d, normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Make sure sqrt isn't negative --> invalid refraction
    bool can_refract = ri * sin_theta <= 1.0;
    if (can_refract)
    {
        float3 r_out_perp =  ri * (ray_d + cos_theta * normal);
        float3 r_out_parallel = -sqrt(fabsf(1.0 - dot(r_out_perp, r_out_perp))) * normal;
        out_dir = r_out_perp + r_out_parallel;
        out_dir = safe_normalize(out_dir);
    }
    return can_refract;
}

static __device__ __inline__ void handleMirror(const float3 ray_d, float3 normal,
                                                float3& new_ray_dir, unsigned int& numBounces)
{
    // Perfect reflection, e.g: ray_d - 2.0 * dot(ray_d, normal) * normal
    float3 reflected_normal = dot(ray_d, normal) < 0.0 ? normal : -normal;
    new_ray_dir = -reflect(ray_d, reflected_normal);
    new_ray_dir = safe_normalize(new_ray_dir);
    numBounces += 1;
}

static __device__ __inline__ void handleGlass(const float3 ray_d, float3 normal,
                                               float3& new_ray_dir,
                                               unsigned int& numBounces, float& hit_t, unsigned int& rndSeed)
{
    const unsigned int triId = optixGetPrimitiveIndex();
    float n1 = 1.0003;
    float n2 = params.refractiveIndex[triId][0];
    float ior = n2 / n1;
    bool is_refracted = refract(new_ray_dir, ray_d, normal, ior, rndSeed); // Updates new_ray_dir if refracted
    if (!is_refracted)
    {
        // Perfect reflection, e.g: ray_d - 2.0 * dot(ray_d, normal) * normal
        float3 reflected_normal = dot(ray_d, normal) < 0.0 ? normal : -normal;
        new_ray_dir = -reflect(ray_d, reflected_normal);
        new_ray_dir = safe_normalize(new_ray_dir);
        numBounces += 1;
    }
    else
    {
        // Move next ray origin a bit forward to avoid repetitive collisions with the same face
        hit_t += REFRACTION_EPS_SHIFT;
    }
}

static __device__ __inline__ void handlePBR(const float3 ray_o, const float3 ray_d, float3 normal,
                                            unsigned int& numBounces, float& hit_t,
                                            float3& new_ray_dir,
                                            unsigned int& rndSeed,
                                            HybridRayPayload* payload)
{
    sampled_cook_torrance_brdf(ray_o, ray_d, hit_t, normal, new_ray_dir, rndSeed, payload);
    new_ray_dir = new_ray_dir / length(new_ray_dir);
}

static __device__ __inline__ void handleDiffuse(const float3 ray_o, const float3 ray_d, float3 normal,
                                                float& hit_t, HybridRayPayload* payload)
{
    // Accumulate all Gaussian particles up to intersection with mesh surface first
    const float4 volumetricRadDns = traceGaussians(*(payload->rayData), ray_o, ray_d, 1e-9, hit_t, payload);
    const float3 volRadiance = make_float3(volumetricRadDns.x, volumetricRadDns.y, volumetricRadDns.z);
    const float volAlpha = volumetricRadDns.w;
    payload->accumulatedColor += volRadiance;
    payload->accumulatedAlpha += volAlpha;

    const float3 diffuse = get_diffuse_color(ray_d, normal);
    const float surfaceAlpha = 1.0 - payload->accumulatedAlpha;
    payload->accumulatedColor += surfaceAlpha * diffuse;
    payload->accumulatedAlpha += surfaceAlpha;
}

static __device__ __inline__ float3 getSmoothNormal()
{
    // Uses interpolated vertex normals to get a smooth varying interpolated normal
    const unsigned int triId = optixGetPrimitiveIndex();
    const unsigned int v0_idx = params.triangles[triId][0];
    const unsigned int v1_idx = params.triangles[triId][1];
    const unsigned int v2_idx = params.triangles[triId][2];
    const float3 n0 = make_float3(params.vNormals[v0_idx][0], params.vNormals[v0_idx][1], params.vNormals[v0_idx][2]);
    const float3 n1 = make_float3(params.vNormals[v1_idx][0], params.vNormals[v1_idx][1], params.vNormals[v1_idx][2]);
    const float3 n2 = make_float3(params.vNormals[v2_idx][0], params.vNormals[v2_idx][1], params.vNormals[v2_idx][2]);
    const float2 barycentric = optixGetTriangleBarycentrics();
    float3 interpolated_normal = (1 - barycentric.x - barycentric.y) * n0 + barycentric.x * n1 + barycentric.y * n2;
    interpolated_normal /= length(interpolated_normal);

    return interpolated_normal;
}

static __device__ __inline__ float3 getHardNormal()
{
    // Computes a "hard" non-varying normal using the vertex positions
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int triId = optixGetPrimitiveIndex();
    const unsigned int gasSbtIdx = optixGetSbtGASIndex();
    float3 v[3] = {};
    optixGetTriangleVertexData(gas, triId, gasSbtIdx, 0, v);
    float3 normal = safe_normalize(cross(v[1] - v[0], v[2] - v[0]));
    return normal;
}

extern "C" __global__ void __closesthit__ch()
{
    // CH is enabled only for primitives tracing by optixTrace flag
    // e.g. guaranteed that:
    // unsigned int next_render_pass = getNextTraceState(); next_render_pass == PGRNDTracePrimitivesPass;

    // Read inputs off payload
    HybridRayPayload* payload = getRayPayload();
    unsigned int numBounces = payload->numBounces;  // Number of times ray was reflected so far
    unsigned int rndSeed = payload->rndSeed;

    const unsigned int triId = optixGetPrimitiveIndex(); // Which face we hit
    float hit_t = optixGetRayTmax();                     // t when ray intersected the surface
    const float3 ray_o = optixGetWorldRayOrigin();       // Ray origin, when ray intersected the surface
    const float3 ray_d = optixGetWorldRayDirection();    // Ray direction, when ray intersected the surface
    // Compute normals using interplated precomputed vertex normals or directly from vertex positions ("non-smooth")
    const float3 normal = (params.playgroundOpts & PGRNDRenderSmoothNormals) ? getSmoothNormal() : getHardNormal();
    auto intersected_type = params.primType[triId][0];   // Primitive type that we hit

    float3 new_ray_dir = make_float3(0.0, 0.0, 0.0);           // Will hold new redirected ray direction
    unsigned int next_render_pass = PGRNDTraceRTGaussiansPass; // Next render pass in most cases is volumetric tracing

    // The following vars will be initialized with new data: new_ray_dir, numBounces, hit_t, rndSeed
    if (intersected_type == PGRNDPrimitiveMirror)
        handleMirror(ray_d, normal, new_ray_dir, numBounces);
    else if (intersected_type == PGRNDPrimitiveGlass)
        handleGlass(ray_d, normal, new_ray_dir, numBounces, hit_t, rndSeed);
    else if (intersected_type == PGRNDPrimitiveDiffuse) {
        handleDiffuse(ray_o, ray_d, normal, hit_t, payload);
        next_render_pass = PGRNDTraceTerminate; // Fully opaque so terminate
    }
    else if (intersected_type == PGRNDPrimitivePBR)
        handlePBR(ray_o, ray_d, normal, numBounces, hit_t, new_ray_dir, rndSeed, payload);
    else
        new_ray_dir = ray_d;    // Do nothing

    // -- Write outputs to payload --
    // Intersection point - also determines origin of next ray
    payload->t_hit = hit_t;
    // If ray has bounces remaining, update next ray orig and dir
    payload->rayOri = ray_o + hit_t * ray_d;
    payload->rayDir = new_ray_dir;
    // Output: Number of times face redirected
    payload->numBounces = numBounces;
    // Update next seed if RNG was used
    payload->rndSeed = rndSeed;

    // Output: Ray hit something so it is considered redirected (->Gaussians pass), or terminate
    setNextTraceState(next_render_pass);
}

extern "C" __global__ void __intersection__is() {
    intersectVolumetricGS();
}

extern "C" __global__ void __anyhit__ah()
{
    // Enabled only for Gaussian ray tracing
    if (getNextTraceState() == PGRNDTraceRTGaussiansPass)
        anyhitSortVolumetricGS();
}

extern "C" __global__ void __miss__ms()
{
    // Ray missed: no primitives - trace remaining Gaussians till bbox end, or terminate
    if (getNextTraceState() == PGRNDTracePrimitivesPass)
    {
        HybridRayPayload* payload = getRayPayload();
        payload->rayMissed = true;
    }
}

#undef __PLAYGROUND__MODE__