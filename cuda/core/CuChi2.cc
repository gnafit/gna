#include <boost/lexical_cast.hpp>

#include "CuChi2.hh"
#include "GNAcuMath.h"
#include <chrono>
#include "cuda_config_vars.h"

using namespace Eigen;

void CuChi2::add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov) {
  t_["cuchi2"].input(theory);
  t_["cuchi2"].input(data);
  t_["cuchi2"].input(cov);
}

void CuChi2::checkTypes(Atypes args, Rtypes rets) {
  if (args.size()%3 != 0) {
    throw args.undefined();
  }
  for (size_t i = 0; i < args.size(); i+=3) {
    if (args[i+0].shape.size() != 1) {
      throw rets.error(rets[0], "non-vector theory");
    }

    if (args[i+1].shape != args[i+0].shape) {
      throw rets.error(rets[0], "data and theory have different shape");
    }
    if (args[i+2].shape.size() != 2 ||
        args[i+2].shape[0] != args[i+2].shape[1]) {
      throw rets.error(rets[0], "incompatible covmat shape");
    }
    if (args[i+2].shape[0] != args[i+0].shape[0]) {
      throw rets.error(rets[0], "incompatible covmat shape 2");
    }
  }
  rets[0] = DataType().points().shape(1);
}

void CuChi2::calculateChi2(Args args, Rets rets) {
  rets[0].arr(0) = 0;
  if (!InvMatCompFlag) {
    InvMatCompFlag = true;
    for (size_t i = 0; i < args.size(); i+=3) {
      if (args[i+2].mat.rows() < CUDA_MAT_SIZE_THRESHOLD) {  // TODO: find opt threshold!
         MatrixXd tmp = MatrixXd::Identity(args[i+2].mat.rows(), args[i+2].mat.cols());
         args[i+2].mat.triangularView<Eigen::Lower>().solveInPlace(tmp);
         L.push_back(MatrixXd(tmp));
      }
      else {
        double * resMat = new double[args[i+2].mat.rows() * args[i+2].mat.rows()];
        cuInverseMat(args[i+2].mat.rows(), 
                   MatrixXd(args[i+2].mat.triangularView<Eigen::Lower>()).data(), 
                   resMat);
        Map<MatrixXd> mat(resMat, args[i+2].mat.rows(), args[i+2].mat.rows());
        L.push_back(mat);
      }
    }
  }
  VectorXd diff;
  for (size_t i = 0, li = 0; i < args.size(); i+=3, li++) {
    diff = args[i+0].vec - args[i+1].vec;
    rets[0].arr(0) += (L[li]*diff).array().square().sum();
  }
}
