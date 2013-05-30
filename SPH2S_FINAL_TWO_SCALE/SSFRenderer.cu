//------------------------------------------------------------------------------
//  SSFRenderer.cpp
//------------------------------------------------------------------------------
#include "SSFRenderer.h"
//------------------------------------------------------------------------------
texture<float, cudaTextureType2D, cudaReadModeElementType> gDepthMap;
texture<float, cudaTextureType2D, cudaReadModeElementType> gIndexMap;
texture<float, cudaTextureType2D, cudaReadModeElementType> gThicknessMap;
//------------------------------------------------------------------------------
__global__ void invert(
    float* dResult,
    unsigned int width,
    unsigned int height
)
{
    unsigned int u = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int v = threadIdx.y + blockDim.y*blockIdx.y;

    if (u >= width || v >= height)
    {
        return;
    }

    unsigned int idx = v*width + u;
    float val = tex2D(gDepthMap, u, v);
    dResult[idx] = val == 0.0f ? 0.0f : 1.0f; 
}
//------------------------------------------------------------------------------
__global__ void blurGaussX(
    float* dResult,
    int blurRadius,
    float blurScale,
    unsigned int width,
    unsigned int height
)
{
    unsigned int u = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int v = threadIdx.y + blockDim.y*blockIdx.y;

    if (u >= width || v >= height)
    {
        return;
    }

    unsigned int idx = v*width + u;
    float res = 0.0f;
    float norm = 0.0f;
    float valc = tex2D(gThicknessMap, u, v);
    
    if (valc == 0.0f)
    {
        dResult[idx] = 0.0f;
        return;
    }

    for (int i = -blurRadius; i <= blurRadius; i++)
    {
        float r = float(i);
        float w = exp(-r*r*blurScale);
        float val = tex2D(gThicknessMap, u + i, v);
        res += val*w;
        norm += w;
    }

    dResult[idx] = res/norm; 
}
//------------------------------------------------------------------------------
__global__ void blurGaussY(
    float* dResult,
    float* dTempResult,
    int blurRadius,
    float blurScale,
    unsigned int width,
    unsigned int height
)
{
    unsigned int u = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int v = threadIdx.y + blockDim.y*blockIdx.y;

    if (u >= width || v >= height)
    {
        return;
    }

    unsigned int idx = v*width + u;
    float res = 0.0f;
    float norm = 0.0f;
    float valc = dTempResult[idx];
    
    if (valc == 0.0f)
    {
        dResult[idx] = 0.0f;
        return;
    }

    for (int i = -blurRadius; i <= blurRadius; i++)
    {
        float r = float(i);
        float w = exp(-r*r*blurScale);
        float val = dTempResult[(v + i)*width + u];
        res += val*w;
        norm += w;
    }

    dResult[idx] = res/norm; 
}
//------------------------------------------------------------------------------
__global__ void blurBilateralX(
    float* dResult,
    int blurRadius,
    float blurScale,
    float blurDepthFalloff,
    unsigned int width,
    unsigned int height
)
{
    unsigned int u = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int v = threadIdx.y + blockDim.y*blockIdx.y;

    if (u >= width || v >= height)
    {
        return;
    }

    unsigned int idx = v*width + u;
    float res = 0.0f;
    float norm = 0.0f;
    float valc = tex2D(gDepthMap, u, v);
    
    if (valc == 0.0f)
    {
        dResult[idx] = 0.0f;
        return;
    }

    for (int i = -blurRadius; i <= blurRadius; i++)
    {
        float val = tex2D(gDepthMap, u + i, v);
        float r = float(i);
        float w = exp(-r*r*blurScale);
        float r2 = (val - valc);
        float g = exp(-r2*r2*blurDepthFalloff);

        res += val*w*g;
        norm += w*g;
    }

    dResult[idx] = res/norm; 
}
//------------------------------------------------------------------------------
__global__ void blurBilateralY(
    float* dResult,
    float* dTempResult,
    int blurRadius,
    float blurScale,
    float blurDepthFalloff,
    unsigned int width,
    unsigned int height
)
{
    unsigned int u = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int v = threadIdx.y + blockDim.y*blockIdx.y;

    if (u >= width || v >= height)
    {
        return;
    }

    unsigned int idx = v*width + u;
    float res = 0.0f;
    float norm = 0.0f;
    float valc = dTempResult[idx];
    
    if (valc == 0.0f)
    {
        dResult[idx] = 0.0f;
        return;
    }

    for (int i = -blurRadius; i <= blurRadius; i++)
    {
        float val = dTempResult[(v + i)*width + u];
        float r = float(i);
        float w = exp(-r*r*blurScale);
        float r2 = (val - valc);
        float g = exp(-r2*r2*blurDepthFalloff);

        res += val*w*g;
        norm += w*g;
    }

    dResult[idx] = res/norm; 
}
//------------------------------------------------------------------------------
__global__ void getSurfaceIDs(
    int* dStates,
    unsigned int width,
    unsigned int height
)
{
    unsigned int u = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int v = threadIdx.y + blockDim.y*blockIdx.y;

    if (u >= width || v >= height)
    {
        return;
    }

    unsigned int id = (unsigned int)tex2D(gIndexMap, u, v);

    if (id == 0)
    {
        return;
    }

    dStates[id - 1] |= 8;

}
//------------------------------------------------------------------------------
static float gsQuadCoords[] = 
{
	0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f
};
//------------------------------------------------------------------------------
static GLuint createProgramDepth(
    const char* vertexShader,
    const char* geometryShader,
    const char* fragmentShader
);
static GLuint createProgramThick(
    const char* vertexShader,
    const char* geometryShader,
    const char* fragmentShader
);
static void createFramebufferObject(
    GLuint& framebufferObject,
    GLuint& renderbufferObject,
    GLuint texture,
    unsigned int width,
    unsigned int height
);
static void createFramebufferObject(
    GLuint& framebufferObject,
    GLuint texture
);
void saveFloatingPointTexturer2DToPPM(
    const char* filename,
    GLuint texture,
    unsigned int width,
    unsigned int height,
    float dontMind
);
void saveIntegerTexture2DToPPM(
    const char* filename,
    GLuint texture,
    unsigned int width,
    unsigned int height
);
//------------------------------------------------------------------------------
SSFRenderer::SSFRenderer(
    ParticleData* dataLow,
    ParticleData* dataHigh,
    unsigned int width, 
    unsigned int height,
    float particleRadius
)
:
    mParticleDataLow(dataLow),
    mParticleDataHigh(dataHigh),
    mParticleRadius(particleRadius),
    mWidth(width), 
    mHeight(height), 
    mBlockDimensions(16, 16, 1)
{
    //--------------------------------------------------------------------------
    // init resources for creating the depth map
    //--------------------------------------------------------------------------

    // create a program that renders the depth map
    mRenderDepthProgram = createProgramDepth(
        "SSFRendererSphereVertex.glsl",
        "SSFRendererSphereGeometry.glsl",
        "SSFRendererDepthFragment.glsl"
    );

    glUseProgram(mRenderDepthProgram);
    GLint loc = glGetUniformLocation(mRenderDepthProgram, "uParticleRadius");
    glUniform1f(loc, particleRadius);
    
    // create the texture for the depth map and register it with CUDA
    GL::CreateFloatingPointTexture2D(
        mDepthTexture, width, height, 1
    );
    CUDA::GL::RegisterImage(
        &mCUDAGraphicsResources[0],
        mDepthTexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsNone
    );

    // create texture for the index map and register it with CUDA
    GL::CreateFloatingPointTexture2D(
        mIndexTexture, width, height, 1
    );
    CUDA::GL::RegisterImage(
        &mCUDAGraphicsResources[1],
        mIndexTexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsNone
    );

    // create a fbo attach depth texture to it 
    glGenFramebuffers(1, &mDepthFramebufferObject);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDepthFramebufferObject);
    
    glFramebufferTexture2D(
        GL_DRAW_FRAMEBUFFER, 
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, 
        mDepthTexture, 
        0
    );

    glFramebufferTexture2D(
        GL_DRAW_FRAMEBUFFER, 
        GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, 
        mIndexTexture, 
        0
    );
	
	glGenRenderbuffers(1, &mDepthRenderbufferObject);
	glBindRenderbuffer(GL_RENDERBUFFER, mDepthRenderbufferObject);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
		GL_RENDERBUFFER, mDepthRenderbufferObject);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "FR N OK " << std::endl;
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    //--------------------------------------------------------------------------
    // init resources for creating the thickness map
    //--------------------------------------------------------------------------

    // create a program that renders the thickness map
    mRenderThicknessProgram = createProgramThick(
        "SSFRendererSphereVertex.glsl",
        "SSFRendererSphereGeometry.glsl",
        "SSFRendererThicknessFragment.glsl"
    );
    glUseProgram(mRenderThicknessProgram);
    loc = glGetUniformLocation(mRenderThicknessProgram, "uParticleRadius");
    glUniform1f(loc, particleRadius);

    // create thickness texture
    GL::CreateFloatingPointTexture2D(
        mThicknessTexture, width, height, 1
    );
    CUDA::GL::RegisterImage(
        &mCUDAGraphicsResources[2],
        mThicknessTexture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsNone
    );

    // create fbo and attach thicknesstexture to it
    createFramebufferObject(
        mThicknessFramebufferObject, 
        mThicknessTexture
    );


    //--------------------------------------------------------------------------
    // init resources for compositing
    //--------------------------------------------------------------------------
    
    // create and initialize program for compositing
    mCompositingProgram = glCreateProgram();
    GL::AttachShader(
        mCompositingProgram, 
        "SSFRendererCompositingVertex.glsl",
        GL_VERTEX_SHADER
    );
    GL::AttachShader(
        mCompositingProgram, 
        "SSFRendererCompositingFragment.glsl",
        GL_FRAGMENT_SHADER
    );
    GL::BindAttribLocation(mCompositingProgram, "inTexCoord", 0);
    GL::BindFragDataLocation(mCompositingProgram, "outFragOutput", 0);
    GL::LinkProgram(mCompositingProgram);
    GL::DumpLog(mCompositingProgram);

    glUseProgram(mCompositingProgram);
    loc = glGetUniformLocation(mCompositingProgram, "uDepthSampler");
    glUniform1i(loc, 0);
    loc = glGetUniformLocation(mCompositingProgram, "uThicknessSampler");
    glUniform1i(loc, 1);
    float texSizeX = 1.0f/static_cast<float>(mWidth);
    float texSizeY = 1.0f/static_cast<float>(mHeight);
    loc = glGetUniformLocation(mCompositingProgram, "uTexSizeX");
    glUniform1f(loc, texSizeX);
    loc = glGetUniformLocation(mCompositingProgram, "uTexSizeY");
    glUniform1f(loc, texSizeY);

    // create vertexarray object for the full screen quad
    glGenVertexArrays(1, &mQuadVertexArrayObject);
    glBindVertexArray(mQuadVertexArrayObject);
    glGenBuffers(1, &mQuadVertexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, mQuadVertexBufferObject);
    glBufferData(
        GL_ARRAY_BUFFER, 
        sizeof(gsQuadCoords), 
        gsQuadCoords,
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    //--------------------------------------------------------------------------
    // Create Vertex array obj for the fluid particles
    //--------------------------------------------------------------------------

    // create vertex array object for the particle data (low)
    glGenVertexArrays(1, &mParticleDataVertexArrayObjects[0]);
    glBindVertexArray(mParticleDataVertexArrayObjects[0]);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleDataLow->PositionsVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleDataLow->StatesVBO);    
    glVertexAttribIPointer(1, 1, GL_INT, 0, 0);
    glEnableVertexAttribArray(1);

    // create vertex array object for the particle data (high)
    glGenVertexArrays(1, &mParticleDataVertexArrayObjects[1]);
    glBindVertexArray(mParticleDataVertexArrayObjects[1]);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleDataHigh->PositionsVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleDataHigh->StatesVBO);
    glVertexAttribIPointer(1, 1, GL_INT, 0, 0);
    glEnableVertexAttribArray(1);

    //--------------------------------------------------------------------------
    //  set up cuda resources for the blurring operations 
    //--------------------------------------------------------------------------

    // allocate temp cuda buffers for smoothing
    CUDA::Alloc<float>(&mdTempData[0], width*height);
    CUDA::Alloc<float>(&mdTempData[1], width*height);

    // compute grid dimensions from block dims and width/height
    mGridDimensions.x = width/mBlockDimensions.x;
    mGridDimensions.y = height/mBlockDimensions.y; 
    mGridDimensions.z = 1; 
    
    // get references to cuda textures
    try
    {
        CUDA_SAFE_INV( 
            cudaGetTextureReference(
                &mTextureReferences[0], 
                &gDepthMap
            ) 
        );
        CUDA_SAFE_INV( 
            cudaGetTextureReference(
                &mTextureReferences[1], 
                &gIndexMap
            ) 
        );
        CUDA_SAFE_INV( 
            cudaGetTextureReference(
                &mTextureReferences[2], 
                &gThicknessMap
            ) 
        );
    }
    catch (const std::runtime_error& error)
    {
        std::cout << error.what() << std::endl;
    }
}
//------------------------------------------------------------------------------
SSFRenderer::~SSFRenderer()
{
    glDeleteVertexArrays(2, mParticleDataVertexArrayObjects);
    glDeleteProgram(mRenderDepthProgram);
    glDeleteTextures(1, &mDepthTexture);
    glDeleteFramebuffers(1, &mDepthFramebufferObject);
    glDeleteProgram(mRenderThicknessProgram);
    glDeleteTextures(1, &mThicknessTexture);
    glDeleteFramebuffers(1, &mThicknessFramebufferObject);
    glDeleteProgram(mCompositingProgram);
    CUDA::Free<float>(&mdTempData[0]);
    CUDA::Free<float>(&mdTempData[1]);
}
//------------------------------------------------------------------------------
void SSFRenderer::Render()
{
    const GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};

    // create depth map
    glBindFramebuffer(GL_FRAMEBUFFER, mDepthFramebufferObject);
    glDrawBuffers(2, buffers);
    glUseProgram(mRenderDepthProgram);
    GLint loc = glGetUniformLocation(mRenderDepthProgram, "uParticleRadius");
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(mParticleDataVertexArrayObjects[0]);
    glUniform1f(loc, mParticleRadius);
    glDrawArrays(GL_POINTS, 0, mParticleDataLow->NumParticles);
    glBindVertexArray(mParticleDataVertexArrayObjects[1]);
    glUniform1f(loc, mParticleRadius*0.5f);
    glDrawArrays(GL_POINTS, 0, mParticleDataHigh->NumParticles);    
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // create thickness map
    glBindFramebuffer(GL_FRAMEBUFFER, mThicknessFramebufferObject);
    glDrawBuffers(1, buffers);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(mRenderThicknessProgram);
    glGetUniformLocation(mRenderThicknessProgram, "uParticleRadius");
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);
    glBindVertexArray(mParticleDataVertexArrayObjects[0]);
    glUniform1f(loc, mParticleRadius);
    glDrawArrays(GL_POINTS, 0, mParticleDataLow->NumParticles);
    glBindVertexArray(mParticleDataVertexArrayObjects[1]);
    glUniform1f(loc, mParticleRadius/2.0f);
    glDrawArrays(GL_POINTS, 0, mParticleDataHigh->NumParticles);
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    try 
    {
        this->blurAndDetectSurface();
    }
    catch (const std::runtime_error& error)
    {
        std::cout << error.what() << std::endl;
        std::system("pause");
    }

    //saveFloatingPointTexturer2DToPPM(
    //    "test3.ppm", 
    //    mThicknessTexture, 
    //    mWidth, 
    //    mHeight, 
    //    0.0f
    //);
    //std::system("pause");

    glUseProgram(mCompositingProgram);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mDepthTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mThicknessTexture);
    glBindVertexArray(mQuadVertexArrayObject);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    //saveFloatingPointTexturer2DToPPM(
    //    "test.ppm", 
    //    mDepthTexture, 
    //    mWidth, 
    //    mHeight, 
    //    0.0f
    //);

    //saveFloatingPointTexturer2DToPPM(
    //    "test3.ppm", 
    //    mThicknessTexture, 
    //    mWidth, 
    //    mHeight, 
    //    0.0f
    //);
    //std::system("pause");
    
}
//------------------------------------------------------------------------------
void SSFRenderer::SetCamera(const GL::Camera& camera)
{
    GLfloat projMat[16];
    GLfloat viewMat[16];
    GL::Camera::ComputeProjectionMatrix(projMat, camera);
    GL::Camera::ComputeViewMatrix(viewMat, camera);
    glUseProgram(mRenderDepthProgram);
    GLint loc = glGetUniformLocation(mRenderDepthProgram, "uProjMat");
    glUniformMatrix4fv(loc, 1, false, projMat);
    loc = glGetUniformLocation(mRenderDepthProgram, "uViewMat");
    glUniformMatrix4fv(loc, 1, false, viewMat);
    glUseProgram(mRenderThicknessProgram);
    loc = glGetUniformLocation(mRenderThicknessProgram, "uProjMat");
    glUniformMatrix4fv(loc, 1, false, projMat);
    loc = glGetUniformLocation(mRenderThicknessProgram, "uViewMat");
    glUniformMatrix4fv(loc, 1, false, viewMat);
    glUseProgram(mCompositingProgram);
    loc = glGetUniformLocation(mCompositingProgram, "uProjMat");
    glUniformMatrix4fv(loc, 1, false, projMat);
}
//------------------------------------------------------------------------------
void SSFRenderer::blurAndDetectSurface()
{
    CUDA_SAFE_INV( cudaGraphicsMapResources(3, mCUDAGraphicsResources) );
    cudaArray* texArray[3];


    //--------------------------------------------------------------------------
    // blur the depth map
    //--------------------------------------------------------------------------

    cudaGraphicsSubResourceGetMappedArray(
        &texArray[0],
        mCUDAGraphicsResources[0],
        0, 0
    );
    cudaChannelFormatDesc desc;
    CUDA_SAFE_INV( cudaGetChannelDesc(&desc, texArray[0]) );
    CUDA_SAFE_INV( 
        cudaBindTextureToArray(
            mTextureReferences[0], 
            texArray[0],
            &desc
        )
    );
    blurBilateralX<<<mGridDimensions, mBlockDimensions>>>(
        mdTempData[0],
        3, 
        0.02f,
        150.0f,
        mWidth,
        mHeight
    );
    blurBilateralY<<<mGridDimensions, mBlockDimensions>>>(
        mdTempData[1],
        mdTempData[0],
        3, 
        0.02f,
        150.0f,
        mWidth,
        mHeight
    );
   
    // copy results to texture
    //CUDA_SAFE_INV( 
    //    cudaMemcpyToArray(
    //        texArray[0],
    //        0, 
    //        0, 
    //        mdTempData[1], 
    //        mWidth*mHeight*sizeof(float), 
    //        cudaMemcpyDeviceToDevice
    //    )
    //);
    cudaUnbindTexture(mTextureReferences[0]);


    //--------------------------------------------------------------------------
    // get surface ids
    //--------------------------------------------------------------------------
    CUDA::Timer timer;
    timer.Start();
    cudaGraphicsSubResourceGetMappedArray(
        &texArray[1],
        mCUDAGraphicsResources[1],
        0, 0
    );
    CUDA_SAFE_INV( 
        cudaBindTextureToArray(
            mTextureReferences[1], 
            texArray[1],
            &desc
        )
    );

    mParticleDataLow->Map();
    getSurfaceIDs<<<mGridDimensions, mBlockDimensions>>>(
        mParticleDataLow->dStates,
        mWidth,
        mHeight
    );
    mParticleDataLow->Unmap();
   
    cudaUnbindTexture(mTextureReferences[1]);
    timer.Stop();
    timer.DumpElapsed();

    //--------------------------------------------------------------------------
    // blur the thickness map
    //--------------------------------------------------------------------------
    
    cudaGraphicsSubResourceGetMappedArray(
        &texArray[2],
        mCUDAGraphicsResources[2],
        0, 0
    );

    CUDA_SAFE_INV( 
        cudaBindTextureToArray(
            mTextureReferences[2], 
            texArray[2],
            &desc
        )
    );
    blurGaussX<<<mGridDimensions, mBlockDimensions>>>(
        mdTempData[0],
        3, 
        0.02f,
        mWidth,
        mHeight
    );
    blurGaussY<<<mGridDimensions, mBlockDimensions>>>(
        mdTempData[1],
        mdTempData[0],
        3, 
        0.02f,
        mWidth,
        mHeight
    );
   
    // copy results to texture
    //CUDA_SAFE_INV( 
    //    cudaMemcpyToArray(
    //        texArray[2],
    //        0, 
    //        0, 
    //        mdTempData[1], 
    //        mWidth*mHeight*sizeof(float), 
    //        cudaMemcpyDeviceToDevice
    //    )
    //);
    cudaUnbindTexture(mTextureReferences[2]);

    cudaGraphicsUnmapResources(3, mCUDAGraphicsResources);
}
//------------------------------------------------------------------------------

//==============================================================================
//  Aux functions here
//==============================================================================

//------------------------------------------------------------------------------
GLuint createProgramDepth(
    const char* vertexShader,
    const char* geometryShader,
    const char* fragmentShader
)
{
    GLuint program = glCreateProgram();
    GL::AttachShader(program, vertexShader, GL_VERTEX_SHADER);
    GL::AttachShader(program, geometryShader, GL_GEOMETRY_SHADER);
    GL::AttachShader(program, fragmentShader, GL_FRAGMENT_SHADER);
    GL::BindAttribLocation(program, "inPosition", 0);
    GL::BindAttribLocation(program, "inState", 1);
    GL::BindFragDataLocation(program, "outFragOutput", 0);
    GL::BindFragDataLocation(program, "outIndex", 1);
    GL::LinkProgram(program);
    GL::DumpLog(program);
    return program;
}
//------------------------------------------------------------------------------
GLuint createProgramThick(
    const char* vertexShader,
    const char* geometryShader,
    const char* fragmentShader
)
{
    GLuint program = glCreateProgram();
    GL::AttachShader(program, vertexShader, GL_VERTEX_SHADER);
    GL::AttachShader(program, geometryShader, GL_GEOMETRY_SHADER);
    GL::AttachShader(program, fragmentShader, GL_FRAGMENT_SHADER);
    GL::BindAttribLocation(program, "inPosition", 0);
    GL::BindAttribLocation(program, "inState", 1);
    GL::BindFragDataLocation(program, "outFragOutput", 0);
    GL::LinkProgram(program);
    GL::DumpLog(program);
    return program;
}
//------------------------------------------------------------------------------
void createFramebufferObject(
    GLuint& framebufferObject,
    GLuint texture
)
{
    glGenFramebuffers(1, &framebufferObject);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebufferObject);

    glFramebufferTexture2D(
        GL_DRAW_FRAMEBUFFER, 
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, 
        texture, 
        0
    );

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}
//------------------------------------------------------------------------------
void createFramebufferObject(
    GLuint& framebufferObject,
    GLuint& renderbufferObject,
    GLuint texture,
    unsigned int width,
    unsigned int height
)
{
    glGenFramebuffers(1, &framebufferObject);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebufferObject);

    glFramebufferTexture2D(
        GL_DRAW_FRAMEBUFFER, 
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, 
        texture, 
        0
    );

	glGenRenderbuffers(1, &renderbufferObject);
	glBindRenderbuffer(GL_RENDERBUFFER, renderbufferObject);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
		GL_RENDERBUFFER, renderbufferObject);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}
//------------------------------------------------------------------------------
void saveFloatingPointTexturer2DToPPM(
    const char* filename,
    GLuint texture,
    unsigned int width,
    unsigned int height,
    float dontMind
)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    unsigned int numData = width*height;
    float* data = new float[numData];
    float* dataFLIP = new float[numData];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data);
    float min = 100000.0f;
    float max = 0.0f;
    
    for (unsigned int i = 0; i < numData; i++)
    {
        float color = data[i];
    
        if (color < 0.0f)
        {
        //    std::cout << color << std::endl;
        }

        if (color > max && color != dontMind)
        {
            max = color;
        }
        if (color < min && color != dontMind)
        {
            min = color;
        }
    }
    min *= 0.95f;
    for (unsigned int i = 0; i < numData; i++)
    {
        data[i] = (data[i] - min)/(max - min);  
    }

    for (unsigned int j = 0; j < height; j++)
    {
        for (unsigned int i = 0; i < width; i++)
        { 
            unsigned int idx = j*width + i;
            unsigned int idxFLIP = (height - 1 - j)*width + i;
            dataFLIP[idxFLIP] = data[idx];
        }
    }
 
    FILE* file = fopen(filename, "w");
    fprintf(file, "P3\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");
    
    for (unsigned int i = 0; i < width*height; i++)
    {
        int pixel[3];
        pixel[0] = static_cast<int>(dataFLIP[i]*255.0f);
        pixel[1] = static_cast<int>(dataFLIP[i]*255.0f);
        pixel[2] = static_cast<int>(dataFLIP[i]*255.0f);
        fprintf(file, "%d %d %d\n", pixel[0], pixel[1], pixel[2]);
    }
    
    fclose(file);
    
    delete[] data;
    delete[] dataFLIP;
    glBindTexture(GL_TEXTURE_2D, 0);
}
//------------------------------------------------------------------------------
void saveIntegerTexture2DToPPM(
    const char* filename,
    GLuint texture,
    unsigned int width,
    unsigned int height
)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    unsigned int numData = width*height;
    float* data = new float[numData];
    float* dataFLIP = new float[numData];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data);

    //for (unsigned int i = 0; i < numData; i++)
    //{
    //    data[i] = (data[i] == 0.0f ? 0 : 255);
    //}

    for (unsigned int j = 0; j < height; j++)
    {
        for (unsigned int i = 0; i < width; i++)
        { 
            unsigned int idx = j*width + i;
            unsigned int idxFLIP = (height - 1 - j)*width + i;
            dataFLIP[idxFLIP] = data[idx];
        }
    }
 
    FILE* file = fopen(filename, "w");
    fprintf(file, "P3\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");
    
    for (unsigned int i = 0; i < width*height; i++)
    {
        float pixel[3];
        pixel[0] = (dataFLIP[i]);
        pixel[1] = (dataFLIP[i]);
        pixel[2] = (dataFLIP[i]);
        fprintf(file, "%f %f %f\n", pixel[0], pixel[1], pixel[2]);
    }
    
    
    delete[] data;
    delete[] dataFLIP;
    glBindTexture(GL_TEXTURE_2D, 0);
}
//------------------------------------------------------------------------------
