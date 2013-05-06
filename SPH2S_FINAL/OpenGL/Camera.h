//-----------------------------------------------------------------------------
#ifndef CAMERA_H
#define CAMERA_H
//-----------------------------------------------------------------------------
#include "Vector3f.h"
//-----------------------------------------------------------------------------
namespace GL
{
//-----------------------------------------------------------------------------
class Camera
{
public:
	Camera (const Vector3f& pos, const Vector3f& foc, const Vector3f& up,
		float aspect, float fovy, float near, float far);
	~Camera ();

	void Rotate (float upAngle, float vAngle);

    void RotateAroundFocus (float dAngX, float dAngY, float dAngZ); 

	static void ComputeViewMatrix (float* mat, const Camera& camera);
	static void ComputeProjectionMatrix (float* mat, const Camera& camera);
    static Vector3f DirectionVector (const Vector3f& pos, const Vector3f& foc);
private:
    void reconfigure ();

	Vector3f mPosition;
	Vector3f mUp;
	Vector3f mFoc;
	Vector3f mV;
    Vector3f mDir;

	float mAspect;
	float mFovY;
	float mNear;
	float mFar;

};
//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
#endif /* end of include guard: CAMERA_H */
//-----------------------------------------------------------------------------
