#ifndef PARAMETRICLAZY_HPP
#define PARAMETRICLAZY_HPP

#include <cmath>

#include <boost/proto/proto.hpp>

#include <boost/fusion/sequence.hpp>
#include <boost/fusion/container.hpp>

#include "dependant.hh"

namespace proto = boost::proto;
namespace mpl = boost::mpl;
namespace fusion = boost::fusion;

template <typename E>
struct parametric_expr;

struct parametric_generator
  : proto::pod_generator<parametric_expr>
{ };

struct parametric_domain
  : proto::domain<parametric_generator, proto::not_<proto::address_of<proto::_> > >
{
  template <typename T>
  struct as_child: proto_base_domain::as_expr<T>
  { };
};

template <typename T>
struct parametric_domain::as_child<parameter<T> >:
  proto_base_domain::as_child<parameter<T> >
{ };

template <typename T>
struct parametric_domain::as_child<const parameter<T>>:
  proto_base_domain::as_child<const parameter<T> >
{ };

template <typename T>
struct parametric_domain::as_child<variable<T> >:
  proto_base_domain::as_child<variable<T> >
{ };

template <typename T>
struct parametric_domain::as_child<const variable<T> >:
  proto_base_domain::as_child<const variable<T> >
{ };

template <typename E>
struct parametric_expr
{
  BOOST_PROTO_EXTENDS(E, parametric_expr<E>, parametric_domain)
};

namespace ParametricLazyOps {
  template <typename T>
  struct is_terminal : mpl::false_ {};

  template <typename T>
  struct is_terminal<parameter<T> > : mpl::true_ { };

  template <typename T>
  struct is_terminal<variable<T> > : mpl::true_ { };

  template <typename T>
  struct is_terminal<dependant<T> > : mpl::true_ { };

  BOOST_PROTO_DEFINE_OPERATORS(is_terminal, parametric_domain)

  const parametric_expr<proto::terminal<double(*)(double, int)>::type > Pow{{std::pow}};
  const parametric_expr<proto::terminal<double(*)(double)>::type > Sqrt{{std::sqrt}};
  const parametric_expr<proto::terminal<double(*)(double)>::type > Sin{{std::sin}};
  const parametric_expr<proto::terminal<double(*)(double)>::type > Cos{{std::cos}};
  const parametric_expr<proto::terminal<double(*)(double, double)>::type > Atan2{{std::atan2}};
}

#define DEFINE_OUTPUT_OPERATOR(type)					\
template <typename ValueType>						\
inline std::ostream &operator<<(std::ostream &sout, type<ValueType> const &v) { \
  if (v.name()) {							\
    return sout << v.name();						\
  } else {								\
    return sout << #type "<>(" << v.rawdata() << ")";			\
  }									\
}

DEFINE_OUTPUT_OPERATOR(parameter)
DEFINE_OUTPUT_OPERATOR(variable)
DEFINE_OUTPUT_OPERATOR(dependant)
#undef DEFINE_OUTPUT_OPERATOR

struct ParametricExpression
  : proto::or_<proto::terminal<parameter<proto::_> >,
	       proto::terminal<dependant<proto::_> >,
	       proto::terminal<double>,
	       proto::nary_expr<proto::_, proto::vararg<ParametricExpression> >
	       >
{};

struct ParametersTransform
  : proto::or_<proto::when<proto::terminal<parameter<proto::_> >, fusion::cons<proto::_value, proto::_state>(proto::_value, proto::_state)>,
               proto::when<proto::terminal<variable<proto::_> >, fusion::cons<proto::_value, proto::_state>(proto::_value, proto::_state)>,
	       proto::when<proto::terminal<dependant<proto::_> >, fusion::cons<proto::_value, proto::_state>(proto::_value, proto::_state)>,
	       proto::when<proto::terminal<proto::_>, proto::_state>,
	       proto::when<proto::binary_expr<proto::_, proto::_, proto::_>, ParametersTransform(proto::_left, ParametersTransform(proto::_right, proto::_state))>
	       >
{ };

template <typename ReturnType, typename Expr, bool Free>
class DependantEvaluable {
  using evaluable_type = dependant<ReturnType>;

public:
  static evaluable_type get(Expr const &e) {
    static std::vector<std::pair<Expr, evaluable_type*> > cached;
    for (std::pair<Expr, evaluable_type*> dep: cached) {
      if (memcmp(&dep.first, &e, sizeof(Expr)) == 0) {
	  return *dep.second;
	}
    }
    std::vector<changeable> deps;
    fusion::for_each(ParametersTransform()(e, fusion::nil()), InconstantsCollector({deps}) );
    auto d = new evaluable_type(*(new DependantEvaluable<ReturnType, Expr, Free>(e)), deps);
    cached.push_back(std::pair<Expr, evaluable_type *>(e, d));
    return *d;
  }
  ReturnType operator()() const{
    proto::default_context ctx;
    return proto::eval(expr, ctx);
  }
protected:
  struct InconstantsCollector {
    template <typename T>
    void operator()(T const &t) const {
      if (std::find(params.begin(), params.end(), t) != params.end()) {
	return;
      }
      params.push_back(t);
    }
    std::vector<changeable> &params;
  };

  DependantEvaluable(Expr const &e): expr(e) { }
  const Expr expr;
};

template <typename Expr>
dependant<double> mkdep_impl(const Expr &expr) {
  return DependantEvaluable<double, Expr, false>::get(expr);
}

template <typename Expr>
dependant<double> mkdep(const Expr &expr) {
  return mkdep_impl(expr);
}

#define PROTIFY(x) (proto::as_child<parametric_domain>(x))

#endif // PARAMETRICLAZY_HPP
