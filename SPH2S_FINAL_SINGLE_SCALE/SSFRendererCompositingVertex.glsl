//------------------------------------------------------------------------------
//  SSFRendererDepthVertex.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform sampler2D uDepthSampler;
uniform sampler2D uThicknessSampler;
uniform sampler2D uSceneSampler;
uniform mat4 uProjMat;
uniform float uTexSizeX;
uniform float uTexSizeZ;
//------------------------------------------------------------------------------
in vec2 inTexCoord;
//------------------------------------------------------------------------------
void main()
{
    vec2 clip;
    clip.x = 2.0f*inTexCoord.x - 1.0f;
    clip.y = 2.0f*inTexCoord.y - 1.0f;
    gl_Position = vec4(clip, 0.0f, 1.0f);
}
//------------------------------------------------------------------------------