//------------------------------------------------------------------------------
//  SceneFragment.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform mat4 uProjMat;
uniform mat4 uViewMat;
//------------------------------------------------------------------------------
in VertexData
{
    smooth vec3 Normal;
}
inVertexData;
//------------------------------------------------------------------------------
out vec4 outFragment;
//------------------------------------------------------------------------------
void main()
{
    vec3 materialColor = vec3(0.9f, 0.9f, 0.9f);
    vec3 ambientColor = vec3(0.9f, 0.9f, 0.9);
    vec3 n = normalize(inVertexData.Normal);
    vec3 l = vec3(0.0f, 0.0f, 1.0f);
    l = normalize(l);

    vec3 outColor = 0.75f*ambientColor + 0.3f*materialColor*max(0.0f, dot(l, n));

    // store color and depth of the scene
    outFragment = vec4(outColor, gl_FragCoord.z);
}
//------------------------------------------------------------------------------
