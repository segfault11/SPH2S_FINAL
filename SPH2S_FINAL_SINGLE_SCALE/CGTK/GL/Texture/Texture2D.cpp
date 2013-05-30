//------------------------------------------------------------------------------
//  Texture2D.cpp
//------------------------------------------------------------------------------
#include "Texture2D.h"
#include <memory.h>
//------------------------------------------------------------------------------
CGTK::GL::Texture2D::Texture2D(
 	GLint internalFormat,
 	GLsizei width,
 	GLsizei height,
 	GLenum format,
    GLenum type
)
{
  	glGenTextures(1, &mGLHandle);
	glBindTexture(GL_TEXTURE_2D, mGLHandle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(
        GL_TEXTURE_2D,
        0, 
        internalFormat, 
        width, 
        height, 
        0, 
        format, 
        type, 
        0
    );    
}
//------------------------------------------------------------------------------
CGTK::GL::Texture2D::~Texture2D()
{
    glDeleteTextures(1, &mGLHandle);
}
//------------------------------------------------------------------------------
GLuint CGTK::GL::Texture2D::GetGLHandle()
{
    return mGLHandle;
}
//------------------------------------------------------------------------------
GLuint CGTK::GL::Texture2D::GetGLHandle() const
{
    return mGLHandle;
}
//------------------------------------------------------------------------------
void CGTK::GL::Texture2D::Bind(GLenum texture)
{
    glActiveTexture(texture);
    glBindTexture(GL_TEXTURE_2D, mGLHandle);
}
//------------------------------------------------------------------------------

