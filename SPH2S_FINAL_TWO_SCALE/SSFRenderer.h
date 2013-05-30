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
        ParticleData* dataLow,
        ParticleData* dataHigh,
        unsigned int width, 
        unsigned int height,
        float particleRadius
    );
    ~SSFRenderer();

    void Render();
    void SetCamera(const GL::Camera& camera);

private:
    void blurAndDetectSurface();

    ParticleData* mParticleDataLow;
    ParticleData* mParticleDataHigh;
    GLuint mParticleDataVertexArrayObjects[2];

    GLuint mRenderDepthProgram;
    GLuint mDepthFramebufferObject;
    GLuint mDepthRenderbufferObject;
    GLuint mDepthTexture;
    GLuint mIndexTexture;

    GLuint mRenderThicknessProgram;
    GLuint mThicknessFramebufferObject;
    GLuint mThicknessTexture;
    float mParticleRadius;
        
    dim3 mGridDimensions;
    dim3 mBlockDimensions;
    const textureReference* mTextureReferences[3];
    float* mdTempData[3];

    GLuint mCompositingProgram;
    GLuint mQuadVertexArrayObject;
    GLuint mQuadVertexBufferObject;

    GLuint mWidth;
    GLuint mHeight;
    cudaGraphicsResource_t mCUDAGraphicsResources[3];
};
//------------------------------------------------------------------------------


