- Vector fundamentals
	- values for Magnitude and direction
	- vector can have difference between 2 point magnitude and direction
	-
- ```cpp
  struct Vector3D
  float
   x, y, z;
  Vector3D(} = default;
  Vector3D(float a, float b, float e)
    {
  X = a;
  y= b;
  z= e;
  } ;
  float& operator [] (int i)
  {
  return ( (&x) [i]);
          }
  const float& operator [] (int i) const
          {
  return ( (&x) [i]);
          }
  Vector3D& operator *=(float s)
           {
  X *= s;
  Y *= s;
  z *= s;
  return (*this);
           }
  Vector3D& operator /=(float s)
           {
  s =l.OF/ s;
  X *= Si
  y *= s;
  z *= s;
  return (*this);
  }
  ```
-
- ```cpp
  inline Vector3D operator *(const Vector3D& v, float s)
  {
  return (Vector3D(v.x * s, v.y * s, v.z * s));
  }
  inline Vector3D operator /(const Vector3D& v, float s)
  {s = l.OF/ s;
  return (Vector3D(v.x * s, v.y * s, v.z * s));
  }
  inline Vector3D operator -(const Vector3D& v)
  {
  return (Vector3D(-v.x, -v.y, -v.z));
  }
  inline float Magnitude(const Vector3D& v)
  {
  return (sqrt(v.x * v.x + v.y * v.y + v.z * v.z));}
  inline Vector3D Normalize(const Vector3D& v)
  {
  return (v / Magnitude(v));}
  ```
-