//------------------------------------------------------------------------------
//  BoxRenderer.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include "OpenGL\OpenGL.h"
#include "OpenGL\Camera.h"
#include "cuda.h"
//------------------------------------------------------------------------------
class BoxRenderer
{
public:
    BoxRenderer (const float3& start, const float3& end);
    ~BoxRenderer ();

    void SetCamera (const GL::Camera& camera);
    void Render () const;

private:
    GLuint mBoxVAO;
    GLuint mBoxVBO;
    GLuint mProgram;
};
//------------------------------------------------------------------------------

