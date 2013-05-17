//------------------------------------------------------------------------------
//  SSFRenderer.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include "OpenGL\OpenGL.h"
#include "OpenGL\Camera.h"
#include "cuda.h"
#include "ParticleData.h"
//------------------------------------------------------------------------------
class SSFRenderer
{
public:
    SSFRenderer(
        const ParticleData* data,
        unsigned int width, 
        unsigned int height,
        float particleRadius
    );
    ~SSFRenderer();

    void Render();
    void SetCamera(const GL::Camera& camera);

private:
    void blur();

    const ParticleData* mParticleData;
    GLuint mParticleDataVertexArrayObject;

    GLuint mRenderDepthProgram;
    GLuint mDepthFramebufferObject;
    GLuint mDepthRenderbufferObject;
    GLuint mDepthTexture;

    GLuint mRenderThicknessProgram;
    GLuint mThicknessFramebufferObject;
    GLuint mThicknessTexture;
    
    dim3 mGridDimensions;
    dim3 mBlockDimensions;
    const textureReference* mTextureReferences[2];
    float* mdTempData[2];

    GLuint mCompositingProgram;
    GLuint mQuadVertexArrayObject;
    GLuint mQuadVertexBufferObject;

    GLuint mWidth;
    GLuint mHeight;
    cudaGraphicsResource_t mCUDAGraphicsResources[2];
};
//------------------------------------------------------------------------------


