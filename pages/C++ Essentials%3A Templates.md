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
- Inheritance( dynamic polymorphisim) and templates(static polymorphisim) dont combine different paradigms. virtual and template dnt mix.
- explicit instantiation function_name<type>();
- preferable to use without explicit instantiation.
- order matters when wanting to define explicitly
- trailing return type
	- ```trailing return
	  auto function() -> int
	  
	  auto function2() -> std::common_type<T,U>::type
	  ```
- typename std::common_type<T,U>::type : return type changes.
- just auto works as return type since c++14 return type deduction
	-
- doesnt work when we have different possible return types.
- use Conditional operator: a<b? b:a . as its an expression. which has a type. by determining the common type.
- SFINAE(substitution Failure is not an error)
- overload template functions
- function template instantiation  or specialization. might have collision in overloads.
	- they are second class citizens template function needs to be considered then branch to specialization if possible . function overloading is better.
- C++20 concept : 
  ```requires
  template<typename T>
  requires std::integral<T> ||std::floating_point<T>
  ```
- vardic templates:
	- ```vardic
	  template< typename T,typename... Ts> //parameter pack
	  void function(int x, const T& value, const Ts&... values)
	  {
	  function(x, values...)//Tail recursion
	  }
	  ```
- packs come last
- ```empty check
  if constexpr(sizeof...(Ts)> 0) //C++17
  ```
- ```fold expressions
  (os<<...<<values);//fold expression(C++17)
  ```
- fold only for operators except ., -> , []
- (...+values);// left to right, doesnt work with 0 parameters
- values+...);// right to left;
-