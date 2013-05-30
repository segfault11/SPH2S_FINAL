//------------------------------------------------------------------------------
//  Framebuffer.h
//------------------------------------------------------------------------------
#include "Framebuffer.h"
#include "..\..\Error\Error.h"
//------------------------------------------------------------------------------
CGTK::GL::Framebuffer::Framebuffer(GLenum target, GLsizei width, GLsizei height)
: 
    mTarget(target),
    mGLDepthHandle(0),
    mWidth(width),
    mHeight(height)
{
    glGenFramebuffers(1, &mGLHandle);
}
//------------------------------------------------------------------------------
CGTK::GL::Framebuffer::~Framebuffer()
{
    glDeleteFramebuffers(1, &mGLHandle);

    if (mGLDepthHandle)
    {
        glDeleteRenderbuffers(1, &mGLDepthHandle);    
    }
}
//------------------------------------------------------------------------------
void CGTK::GL::Framebuffer::Bind()
{
    glBindFramebuffer(mTarget, mGLHandle);
}
//------------------------------------------------------------------------------
void CGTK::GL::Framebuffer::Unbind()
{
    glBindFramebuffer(mTarget, 0);
}
//------------------------------------------------------------------------------
void CGTK::GL::Framebuffer::AttachTexture2D(
    const CGTK::GL::Texture2D& texture, 
    GLenum attachment
)
{
    mDrawBuffers.push_back(attachment);
    Bind();
    glFramebufferTexture2D(
        mTarget, 
        attachment,
        GL_TEXTURE_2D, 
        texture.GetGLHandle(), 
        0
    );
    Unbind();
}
//------------------------------------------------------------------------------
void CGTK::GL::Framebuffer::AttachDepthComponent()
{
    if (mGLDepthHandle)
    {
        return;
    }

    Bind();
    glGenRenderbuffers(1, &mGLDepthHandle);
    glBindRenderbuffer(GL_RENDERBUFFER, mGLDepthHandle);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mWidth, mHeight);
	glFramebufferRenderbuffer(
        mTarget, 
        GL_DEPTH_ATTACHMENT, 
		GL_RENDERBUFFER, 
        mGLDepthHandle
    );
    Unbind();
}
//------------------------------------------------------------------------------
void CGTK::GL::Framebuffer::RegisterAttachments()
{
    Bind();
    glDrawBuffers(mDrawBuffers.size(), mDrawBuffers.data());

    if (glCheckFramebufferStatus(mTarget) != GL_FRAMEBUFFER_COMPLETE)
    {
        CGTK::Error::ReportError("Framebuffer is incomplete");
    }

    Unbind();
}
//------------------------------------------------------------------------------
