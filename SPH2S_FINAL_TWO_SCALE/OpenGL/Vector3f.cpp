//-----------------------------------------------------------------------------
#include "Vector3f.h"
#include <cmath>
#include <iostream>
//-----------------------------------------------------------------------------
GL::Vector3f::Vector3f ()
: mX(0.0f), mY(0.0f), mZ(0.0f)
{

}
//-----------------------------------------------------------------------------
GL::Vector3f::Vector3f (float x, float y, float z)
: mX(x), mY(y), mZ(z)
{

}
//-----------------------------------------------------------------------------
GL::Vector3f::~Vector3f ()
{

}
//-----------------------------------------------------------------------------
void GL::Vector3f::Normalize ()
{
	float mag = std::sqrt(mX*mX + mY*mY + mZ*mZ);
	mX /= mag;
	mY /= mag;
	mZ /= mag;
}
//-----------------------------------------------------------------------------
void GL::Vector3f::ComputeCrossProduct
(
	Vector3f& res, 
	const Vector3f& a, 
	const Vector3f& b
)
{
	res.mX = a.mY*b.mZ - a.mZ*b.mY;
	res.mY = a.mZ*b.mX - a.mX*b.mZ;
	res.mZ = a.mX*b.mY - a.mY*b.mX;
}
//-----------------------------------------------------------------------------
void GL::Vector3f::ComputeDotProduct
(
	float& res,
	const Vector3f& a,
	const Vector3f& b
)
{
	res = a.mX*b.mX + a.mY*b.mY + a.mZ*b.mZ; 
}
//-----------------------------------------------------------------------------
float GL::Vector3f::ComputeNorm (const Vector3f& a)
{
    return std::sqrt(a.mX*a.mX + a.mY*a.mY + a.mZ*a.mZ);
}
//-----------------------------------------------------------------------------
void GL::Vector3f::Substract
(
	Vector3f& res, 
	const Vector3f& a, 
	const Vector3f& b
)
{
	res.mX = a.mX - b.mX;
	res.mY = a.mY - b.mY;
	res.mZ = a.mZ - b.mZ;
}
//-----------------------------------------------------------------------------
void GL::Vector3f::Add
(
	Vector3f& res, 
	const Vector3f& a, 
	const Vector3f& b
)
{
	res.mX = a.mX + b.mX;
	res.mY = a.mY + b.mY;
	res.mZ = a.mZ + b.mZ;
}
//-----------------------------------------------------------------------------
void GL::Vector3f::operator*= (float s)
{
	mX *= s;
	mY *= s;
	mZ *= s;
}
//-----------------------------------------------------------------------------
void GL::Vector3f::operator+= (const GL::Vector3f& a)
{
	mX += a.mX;
	mY += a.mY;
	mZ += a.mZ;
}
//-----------------------------------------------------------------------------
void GL::Vector3f::Dump () const
{
	std::cout << "[" << mX << " " << mY << " " << mZ << "]" << std::endl;
}
//-----------------------------------------------------------------------------
void GL::Vector3f::Rotate (const Vector3f& axis, float angle)
{
	Vector3f k(axis);
	k.Normalize();
	Vector3f kxv;
	ComputeCrossProduct(kxv, k, *this);
	float dp;
	ComputeDotProduct(dp, k, *this);

	// cf. Rodrigues Rotation Equation
	mX = mX*cos(angle) + kxv.mX*sin(angle) + k.mX*dp*(1.0f - cos(angle));
	mY = mY*cos(angle) + kxv.mY*sin(angle) + k.mY*dp*(1.0f - cos(angle));
	mZ = mZ*cos(angle) + kxv.mZ*sin(angle) + k.mZ*dp*(1.0f - cos(angle));
}
//-----------------------------------------------------------------------------
