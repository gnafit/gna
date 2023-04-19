#include <boost/math/constants/constants.hpp>
#include "HistSmear.hh"
#include "TypesFunctions.hh"
#include "fmt/format.h"

#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#include "DataLocation.hh"
#endif

using GNA::SquareMatrixType;

HistSmear::HistSmear(SquareMatrixType matrix_type) {
  bool upper = matrix_type==SquareMatrixType::UpperTriangular;
  transformation_("smear")
      .input("Ntrue")
      .input("SmearMatrix")
      .output("Nrec")
      .types(TypesFunctions::if1d<0>, TypesFunctions::ifSquare<1>,  TypesFunctions::pass<0,0>)
      .types([](TypesFunctionArgs fargs) {
               auto& args=fargs.args;
               if (args[1].shape[0] != args[0].shape[0]) {
                   auto msg = fmt::format("SmearMatrix is not consistent with data vector: {0}x{1} vs {2}", args[1].shape[0], args[1].shape[1], args[0].shape[0] );
                   throw args.error(args[0], msg);
               }
             })
       .func( upper ? &HistSmear::calcSmearUpper : &HistSmear::calcSmear )
#ifdef GNA_CUDA_SUPPORT
       .func("gpu", upper ? &HistSmear::calcSmearUpper_gpu : &HistSmear::calcSmear_gpu, DataLocation::Device)
#endif
	;
}

void HistSmear::calcSmearUpper(FunctionArgs fargs) {
  auto& args=fargs.args;
  fargs.rets[0].x = args[1].mat.triangularView<Eigen::Upper>() * args[0].vec;
}

void HistSmear::calcSmear(FunctionArgs fargs) {
  auto& args=fargs.args;
  fargs.rets[0].x = args[1].mat * args[0].vec;
}

#ifdef GNA_CUDA_SUPPORT

#endif
