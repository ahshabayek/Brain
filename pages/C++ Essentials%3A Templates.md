- -O0 all optimization off
- Overloading v.s Template:
	- same behavior different types:template
	- overload different types different implementations.
	- templates sre a compile time/ static polymorphiosim
	- dynamic polymorphsim
- one Definition Rule:
	- uses the first encountered definition
- class vs typename are the same although class preceded it.
- template is not a function/ class yet
	- template instantiation is needed.
	- ```template declartion
	  template <typename T> // temeplate argument deduction
	  T max(T a, T b)
	  {
	  
	  }
	  ```
- Linker is responsible to deduct how many instantiations are needed.
- wandbox.org
- godbolt.org
- ```template with overload
  struct Widget{
  int i;
  }
  bool operator<(Widget a, Widget b)
  {
  return a.i < b.i;
  
  }
  ```
- this work with