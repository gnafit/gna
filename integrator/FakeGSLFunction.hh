#ifndef FakeGSLFunction_H
#define FakeGSLFunction_H

#include <gsl/gsl_integration.h>

/* The original problem was how to pass a class member function to GSL
 * integration routine, hopefully this will work.
 * The solution is found at StackOverflow
 * http://stackoverflow.com/questions/13074756/how-to-avoid-static-member-function-when-using-gsl-with-c/18181494#18181494 */

template< typename F >
  class gsl_function_pp : public gsl_function {
  public:
  gsl_function_pp(const F& func) : _func(func) {
    function = &gsl_function_pp::invoke;
    params=this;
  }
  private:
  const F& _func;
  static double invoke(double x, void *params) {
    return static_cast<gsl_function_pp*>(params)->_func(x);
  }
};

/* Usage case :
 *  Class* ptr2 = this;
 *  auto ptr = [=](double x)->double{return ptr2->foo(x);};
 *  gsl_function_pp<decltype(ptr)> Fp(ptr);     
 *  gsl_function *F = static_cast<gsl_function*>(&Fp);   */
/* Note that declaring every member function as static should also work as well, since static function location in memory is known even before instantiation of object of class */


#endif
