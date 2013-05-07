//------------------------------------------------------------------------------
//  RendererFragment.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform mat4 uViewMat;
uniform mat4 uProjMat;
uniform vec3 uLightDir;
uniform float uAmbientCoefficient;
uniform float uDiffuseCoefficient;
uniform float uSpecularCoefficient;
uniform float uParticleRadius;
//------------------------------------------------------------------------------
in GeometryData 
{
	vec2 TexCoord;
    vec4 PosView;
} 
inGeometryData;
//------------------------------------------------------------------------------
out vec4 oFragColor;
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

    n.z = -sqrt(1.0f - mag2);

    // compute view space pos for this fragment
    vec4 posView = inGeometryData.PosView + uParticleRadius*vec4(n, 1.0f);

    // compute color for this fragment
    vec3 colorLight = vec3(0.0f, 0.3f, 1.0f);
    vec3 v = vec3(posView.x, posView.y, posView.z);
    vec3 h = v + uLightDir;
    normalize(h);
    normalize(v);

    vec3 color = (uAmbientCoefficient + uDiffuseCoefficient*dot(uLightDir, n) + uSpecularCoefficient*pow(dot(h,n), 3))*colorLight;

    oFragColor = vec4(color, 1.0f);
}
//------------------------------------------------------------------------------
