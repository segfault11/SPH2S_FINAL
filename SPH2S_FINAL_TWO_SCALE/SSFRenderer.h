//------------------------------------------------------------------------------
//  SSFRenderer.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include "OpenGL\OpenGL.h"
#include "OpenGL\Camera.h"
#include "cuda.h"
#include "ParticleData.h"
#include "CGTK\GL\Texture\Texture2D.h"
#include "CGTK\GL\Framebuffer\Framebuffer.h"
//------------------------------------------------------------------------------
class SSFRenderer
{
    class Geometry
    {
        // class that stores the geometry of an object in the scene and is
        // able to render it.
    
    public:
        Geometry(        
            const float* vertices, 
            const float* normals,
            unsigned int numFaces
        );
        ~Geometry();
        void Render() const;

        static Geometry* CreateBox(
            const float3& startPoint, 
            const float3& endPoint
        );

    private:
        GLuint mVAO;
        GLuint mPositionsVBO;
        GLuint mNormalsVBO;
        GLsizei mNumFaces;
    };

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

    // adds a box to the scence
    void AddBox(    
        const float3& startPoint, 
        const float3& endPoint
    );

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

    GLuint mSceneProgram;
    std::vector<Geometry*> mSceneGeometry;
    CGTK::GL::Framebuffer* mSceneFramebuffer;
    CGTK::GL::Texture2D* mSceneTexture;
    
    GLuint mWidth;
    GLuint mHeight;
    cudaGraphicsResource_t mCUDAGraphicsResources[3];
};
//------------------------------------------------------------------------------


