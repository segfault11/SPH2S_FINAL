//------------------------------------------------------------------------------
//  SSFRenderer.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include "OpenGL\OpenGL.h"
#include "OpenGL\Camera.h"
#include "cuda.h"
#include "ParticleData.h"
#include <vector>
#include "CGTK\GL\Texture\Texture2D.h"
#include "CGTK\GL\Framebuffer\Framebuffer.h"
//------------------------------------------------------------------------------
class SSFRenderer
{
    //  This class performs screen space fluid rendering for the particle 
    //  particle data. Further geometry can be added to the scene in form of
    //  vertex lists. 
    //
    //  TODO:
    //  - let user configure material of fluid and geometry.
    //  - let user configure lighting parameters.
    //  - let user configure smoothing parameters.

private:
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
        const ParticleData* data,
        unsigned int width, 
        unsigned int height,
        float particleRadius
    );
    ~SSFRenderer();

    void Render();
    void SetCamera(
        const GL::Camera& camera
    );

    // adds a box to the scence
    void AddBox(    
        const float3& startPoint, 
        const float3& endPoint
    );

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

    GLuint mSceneProgram;
    std::vector<Geometry*> mSceneGeometry;
    CGTK::GL::Framebuffer* mSceneFramebuffer;
    CGTK::GL::Texture2D* mSceneTexture;

    GLuint mWidth;
    GLuint mHeight;
    cudaGraphicsResource_t mCUDAGraphicsResources[2];
};
//------------------------------------------------------------------------------