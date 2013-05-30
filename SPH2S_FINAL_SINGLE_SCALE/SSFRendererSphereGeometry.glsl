//------------------------------------------------------------------------------
//  SSFRendererSphereGeometry.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform mat4 uViewMat;
uniform mat4 uProjMat;
uniform float uParticleRadius;
//------------------------------------------------------------------------------
layout (points) in;
in VertexData 
{
    float SphereRadiusX;
    float SphereRadiusY;
    vec4 PosView;
}
inVertexData[1];
//------------------------------------------------------------------------------
layout (triangle_strip, max_vertices = 4) out;
out GeometryData 
{
	vec2 TexCoord;  // Tex coords of the created squad 
    vec4 PosView;
} 
outGeometryData;
//------------------------------------------------------------------------------
void main ()
{
    float dx = inVertexData[0].SphereRadiusX;
	float dy = inVertexData[0].SphereRadiusY;

    gl_Position = gl_in[0].gl_Position + vec4(+dx, +dy, 0.0f, 0.0f);
	outGeometryData.TexCoord = vec2(1.0f, 1.0f);
    outGeometryData.PosView = inVertexData[0].PosView;
	EmitVertex();

    gl_Position = gl_in[0].gl_Position + vec4(-dx, +dy, 0.0f, 0.0f);
	outGeometryData.TexCoord = vec2(0.0f, 1.0f);
    outGeometryData.PosView = inVertexData[0].PosView;
	EmitVertex();

    gl_Position = gl_in[0].gl_Position + vec4(+dx, -dy, 0.0f, 0.0f);
	outGeometryData.TexCoord = vec2(1.0f, 0.0f);
    outGeometryData.PosView = inVertexData[0].PosView;
	EmitVertex();

    gl_Position = gl_in[0].gl_Position + vec4(-dx, -dy, 0.0f, 0.0f);
	outGeometryData.TexCoord = vec2(0.0f, 0.0f);
    outGeometryData.PosView = inVertexData[0].PosView;
	EmitVertex();

    EndPrimitive();
}
//-----------------------------------------------------------------------------