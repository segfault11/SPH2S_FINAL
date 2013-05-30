//------------------------------------------------------------------------------
//  Texture2D.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include <gl\glew.h>
#include "..\..\CGTK.h"
//------------------------------------------------------------------------------
namespace CGTK {
namespace GL {
//------------------------------------------------------------------------------
class Texture2D
{
public:
    Texture2D(
 	    GLint internalFormat,
 	    GLsizei width,
 	    GLsizei height,
 	    GLenum format,
 	    GLenum type
    );

    ~Texture2D();

    // binds the handle to a opengl texture unit. the texture is bound to the 
    // unit until [Unbind] is called or another texture is bound to the same
    // unit.
    void Bind(GLenum texture);
    
    // returns the opengl handle
    GLuint GetGLHandle();
    GLuint GetGLHandle() const;
private:
    CGTK_NON_COPYABLE(Texture2D)

    GLuint mGLHandle;
};
//------------------------------------------------------------------------------
} // end of namespace GL
} // end of namespace CGTK 
//------------------------------------------------------------------------------
