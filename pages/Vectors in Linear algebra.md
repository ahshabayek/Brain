- [[Linear Algebra]]
-
- Vector is an ordered array
- each entry is a component.
- Components  are Indexed starting 0
- Belongs to real numbers
- Has direction and length starting from origin point.
- Length is euclidean
- Unit Basis Vectors(Standard basis vectors)
	- of size n of zeros except 1 component which is the indicated index one e.g: e of 3 , third is 1.
	- Generalizes notation to arbitrary length vectors.
- 2 vectors are equal only when all of their elements are equal
- Vectors dont have a spacial location. therefore heeling vectors is possible (heel to toe)
- black vector is heel to toe addition of 2 vectors. choice of which is heel and toe doesn't matter.
- scaling is same as stretching a vector.
- scaled vectors have same direction but different length.
- vector subtraction is the opposite diagonal to vector addition
- ![image.png](../assets/image_1669070858511_0.png)
- AXPY(scaled vector addition) scalar Alpha time X plus Y  : ax+y
- vector memops for axpy, 1 memop for multiplier, 3 * n memops (2 read 1 write), 2*n floating ops.
- Linear combination of vectors scales the individual vectors then adds them
- Dot or inner product.
  multiply each element by it corresponding vector element and add each consecutive multiplication(summation of element multiplication)
- dot product is commutative.
- Euclidean length is root of sum of squares of components
  xT.x
- Law of cosines: 
  ![{\displaystyle a^{2}=b^{2}+c^{2}-2bc\cos \alpha ,}](https://wikimedia.org/api/rest_v1/media/math/render/svg/97e113fcfdbace3e4f6e1204ead3db64ebadd74f)
- A vector function is  a math of one or more scalar and or vector whose output is a vector.
- slicing and dicing dot product: you can partition vectors to sub vectors while carrying out the dot product and then do a summation.