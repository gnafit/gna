#include <boost/math/constants/constants.hpp>
#include "HistSmear.hh"
#include "TypesFunctions.hh"

#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#include "DataLocation.hh"
#endif


HistSmear::HistSmear(bool upper) {
  transformation_("smear")
      .input("Ntrue")
      .input("SmearMatrix")
      .output("Nrec")
      .types(TypesFunctions::if1d<0>, TypesFunctions::ifSquare<1>,  TypesFunctions::pass<0,0>)
      .types([](TypesFunctionArgs fargs) {
               auto& args=fargs.args;
               if (args[1].shape[0] != args[0].shape[0]) {
                 throw args.error(args[0], "SmearMatrix is not consistent with data vector");
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

void HistSmear::calcSmearUpper_gpu(FunctionArgs fargs) {
  auto& args=fargs.args;
  fargs.rets[0].x = args[1].mat.triangularView<Eigen::Upper>() * args[0].vec;
//TODO triangular view in cuda

}

void HistSmear::calcSmear(FunctionArgs fargs) {
  auto& args=fargs.args;
  fargs.rets[0].x = args[1].mat * args[0].vec;
}

#ifdef GNA_CUDA_SUPPORT
void HistSmear::calcSmear_gpu(FunctionArgs fargs) {
  fargs.args.touch();
  auto& gpuargs=fargs.gpu;
  //fargs.rets[0].x = args[1].mat * args[0].vec;
  cuproduct_mat2vec(gpuargs->args, gpuargs->rets, fargs.args[1].mat.cols(), fargs.args[1].mat.rows());
}
#endif
