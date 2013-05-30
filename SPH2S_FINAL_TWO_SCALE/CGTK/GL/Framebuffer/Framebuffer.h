//------------------------------------------------------------------------------
//  Framebuffer.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include <gl\glew.h>
#include "..\..\CGTK.h"
#include "..\Texture\Texture2D.h"
#include <vector>
//------------------------------------------------------------------------------
namespace CGTK {
namespace GL {
//------------------------------------------------------------------------------
class Framebuffer
{
public:
    Framebuffer(GLenum target, GLsizei width, GLsizei height);
    ~Framebuffer();

    void Bind();
    void Unbind();
    void AttachTexture2D(
        const CGTK::GL::Texture2D& texture, 
        GLenum attachment
    );
    void AttachDepthComponent();
    void RegisterAttachments();
    GLuint GetGLHandle();

private:
    CGTK_NON_COPYABLE(Framebuffer)
    
    GLenum mTarget;
    GLuint mGLHandle;
    GLuint mGLDepthHandle;
    GLsizei mWidth;
    GLsizei mHeight;
    std::vector<GLenum> mDrawBuffers;
};
//------------------------------------------------------------------------------
} // end of namespace GL
} // end of namespace CGTK 
//------------------------------------------------------------------------------
