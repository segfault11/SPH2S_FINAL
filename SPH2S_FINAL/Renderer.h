//------------------------------------------------------------------------------
//  Renderer.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include "ParticleData.h"
#include "OpenGL\Camera.h"
//------------------------------------------------------------------------------
class RendererConfig
{
public:
    RendererConfig (
        float3 lightDir,
        float ambientCoefficient,
        float diffuseCoefficient,
        float specularCoefficient,
        float particleRadius
    )
    :
        LightDir(lightDir),
        AmbientCoefficient(ambientCoefficient),
        DiffuseCoefficient(diffuseCoefficient),
        SpecularCoefficient(specularCoefficient),
        ParticleRadius(particleRadius)
    {
    
    }

    float3 LightDir;
    float AmbientCoefficient;
    float DiffuseCoefficient;
    float SpecularCoefficient;
    float ParticleRadius;
};
//------------------------------------------------------------------------------
class Renderer
{
public:
    Renderer (const ParticleData* data, const RendererConfig& config);
    ~Renderer ();

    void SetCamera (const GL::Camera& camera);
    void Render () const;

private:
    GLuint mParticleDataVAO;
    GLuint mProgram;

    const ParticleData* mParticleData;
};
//------------------------------------------------------------------------------
