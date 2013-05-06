//------------------------------------------------------------------------------
//  Renderer.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include "ParticleData.h"
#include "OpenGL\Camera.h"
//------------------------------------------------------------------------------
class Renderer
{
public:
    Renderer (const ParticleData* data);
    ~Renderer ();

    void SetCamera (const GL::Camera& camera);
    void Render () const;

private:
    GLuint mParticleDataVAO;
    GLuint mProgram;

    const ParticleData* mParticleData;
};
//------------------------------------------------------------------------------
