//------------------------------------------------------------------------------
//  RendererFragment.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform mat4 uViewMat;
uniform mat4 uProjMat;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform float uAmbientCoefficient;
uniform float uDiffuseCoefficient;
uniform float uSpecularCoefficient;
uniform float uParticleRadius;
//------------------------------------------------------------------------------
in GeometryData 
{
	vec2 TexCoord;
    vec4 PosView;
    float ColorValue;
} 
inGeometryData;
//------------------------------------------------------------------------------
out vec4 oFragColor;
//------------------------------------------------------------------------------
vec3 getJetColor(float value) 
{
     float fourValue = 4 * value;
     float red   = min(fourValue - 1.5, -fourValue + 4.5);
     float green = min(fourValue - 0.5, -fourValue + 3.5);
     float blue  = min(fourValue + 0.5, -fourValue + 2.5);
 
     return clamp( vec3(red, green, blue), 0.0, 1.0 );
}
//------------------------------------------------------------------------------
void main ()
{
    vec3 n;
    n.x = 2.0f*inGeometryData.TexCoord.x - 1.0f;
    n.y = 2.0f*inGeometryData.TexCoord.y - 1.0f;
    float mag2 = dot(n.xy, n.xy);
    //
    //if (inGeometryData.PosView.z > -1.7f)
    //{
    //    discard;
    //}


    if (mag2 > 1.0f)
    {
        discard;
    }

    n.z = sqrt(1.0f - mag2);

    // compute view space pos for this fragment
    vec4 posView = inGeometryData.PosView + uParticleRadius*vec4(n, 1.0f);

    // compute color for this fragment
    vec3 v = vec3(posView.x, posView.y, posView.z);
    vec3 h = v + uLightDir;
    normalize(h);
    normalize(v);

    vec3 color = (uAmbientCoefficient + uDiffuseCoefficient*dot(uLightDir, n) 
        + uSpecularCoefficient*pow(dot(h,n), 3))*
        getJetColor(inGeometryData.ColorValue);

    oFragColor = vec4(color, 1.0f);


    vec4 posClip = uProjMat*posView;

    gl_FragDepth = posClip.z/posClip.w;
}
//------------------------------------------------------------------------------
