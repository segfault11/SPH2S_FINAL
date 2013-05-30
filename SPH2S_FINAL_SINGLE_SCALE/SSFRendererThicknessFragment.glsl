//------------------------------------------------------------------------------
//  SSFRendererThicknessFragment.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform mat4 uViewMat;
uniform mat4 uProjMat;
uniform float uParticleRadius;
uniform float uScreenWidth;
uniform float uScreenHeight;
uniform sampler2D uSceneSampler;
//------------------------------------------------------------------------------
in GeometryData 
{
	vec2 TexCoord;
    vec4 PosView;
} 
inGeometryData;
//------------------------------------------------------------------------------
out vec4 outFragOutput;
//------------------------------------------------------------------------------
void main ()
{
    vec3 n;
    n.x = 2.0f*inGeometryData.TexCoord.x - 1.0f;
    n.y = 2.0f*inGeometryData.TexCoord.y - 1.0f;
    float mag2 = dot(n.xy, n.xy);
    
    if (mag2 > 1.0f)
    {
        discard;
    }

    n.z = sqrt(1.0f - mag2);

    // compute view space pos for this fragment
    vec4 posView = inGeometryData.PosView + uParticleRadius*vec4(n, 1.0f);

    // compute the depth of this fragment
    vec4 posClip = uProjMat*posView;
    float depth = 0.5f*(posClip.z/posClip.w + 1.0f);

    // compute [0, 1] range position of the fragment to look up the depth of
    // the background scene
    vec2 texCoord;
    texCoord.x = gl_FragCoord.x/uScreenWidth;
    texCoord.y = gl_FragCoord.y/uScreenHeight;
    float sceneDepth = texture(uSceneSampler, texCoord).w;

    // if the the scene depth is smaller than the particle depth than this
    // particle does not contribute to the thickness as it lies behind the 
    // sence. hence, it is discarded.
    if (depth > sceneDepth)
    {
        discard;
    }

    outFragOutput = vec4(2.0f*uParticleRadius, 0.0f, 0.0f, 0.0f);
}
//------------------------------------------------------------------------------