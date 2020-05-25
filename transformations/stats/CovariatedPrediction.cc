#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include "CovariatedPrediction.hh"

using Segment = CovariatedPrediction::Segment;

CovariatedPrediction::CovariatedPrediction()
  : m_finalized(false)
{
  transformation_("prediction")
    .output("prediction")
    .types(&CovariatedPrediction::calculateTypes)
    .func(&CovariatedPrediction::calculatePrediction)
    ;
  transformation_("covbase")
    .output("covbase")
    .types(&CovariatedPrediction::calculateCovbaseTypes)
    .func(&CovariatedPrediction::calculateCovbase)
    ;
  transformation_("cov")
    .input("covbase")
    .output("L")
    .types(&CovariatedPrediction::calculateCovTypes)
    .func(&CovariatedPrediction::calculateCov)
    ;
}

//CovariatedPrediction::CovariatedPrediction(const CovariatedPrediction &other) :
    //m_inputs(other.m_inputs), m_finalized(other.m_finalized), m_prediction_ready(other.m_prediction_ready)
//{
//}

//CovariatedPrediction& CovariatedPrediction::operator=(const CovariatedPrediction &other) {
  //m_inputs = other.m_inputs;
  //m_finalized = other.m_finalized;
  //m_prediction_ready = other.m_prediction_ready;
  //return *this;
//}

/**
   * @brief Add observable into prediction transformation.
   * @param obs -- observable to connect to prediction.
*/
void CovariatedPrediction::append(SingleOutput& obs) {
  if (m_finalized) {
    throw std::runtime_error("appending to finalized CovariatedPrediction");
  }
  t_["prediction"].input(obs);
  m_inputs.push_back(obs.single());
}

/**
   * @brief Finalize the creation of covariances and prediction. Revaulates
   * types and forbids adding new inputs to prediction and covariations.
*/

void CovariatedPrediction::prediction_ready() {
  m_prediction_ready = true;
  t_["prediction"].updateTypes();
}

void CovariatedPrediction::finalize() {
  m_finalized = true;
  t_["prediction"].updateTypes();
  t_["covbase"].updateTypes();
  t_["cov"].updateTypes();
}

/**
   * @brief Given an OutputDescriptor finds its' index in inputs of
   * CovariatedPrediction
   * if present, else throws std::runtime_error
   * @param inp -- The OutputDescriptor for which we need to find index
*/
size_t CovariatedPrediction::blockOffset(OutputDescriptor inp) {
  for (size_t i = 0; i < m_inputs.size(); ++i) {
    if (m_inputs[i].rawptr() == inp.rawptr()) {
      return i;
    }
  }
  throw std::runtime_error("can't find required prediction");
}

/**
   * @brief Returns a number of inputs of CovariatedPrediction
*/
size_t CovariatedPrediction::blocksCount() const noexcept {
  return m_inputs.size();
}


void CovariatedPrediction::covariate(SingleOutput &cov,
                                     SingleOutput &obs1, size_t n1,
                                     SingleOutput &obs2, size_t n2) {
  if (m_finalized) {
    throw std::runtime_error("prediction is finalized");
  }
  size_t idx1 = blockOffset(obs1.single());
  size_t idx2 = blockOffset(obs2.single());
  if (idx1 == idx2 && n1 == n2) {
    CovarianceAction act(CovarianceAction::Diagonal);
    act.a = Segment{idx1, n1};
    t_["covbase"].input(cov);
    m_covactions.push_back(act);
  } else {
    CovarianceAction act(CovarianceAction::Block);
    act.a = Segment{idx1, n1};
    act.b = Segment{idx2, n2};
    if (act.a->i < act.b->i) {
      std::swap(act.a, act.b);
    }
    if (act.b->i + act.b->n > act.a->i) {
      throw std::runtime_error("overlapping covariance");
    }
    t_["covbase"].input(cov);
    m_covactions.push_back(act);
  }
}




/**
   * @brief Add input vector to be used for rank-1 update of covariance matrix
   * @param vec -- vector to be added into inputs of "cov" transofrmation
*/
void CovariatedPrediction::rank1(SingleOutput& vec) {
  t_["cov"].input(vec);
}

void CovariatedPrediction::addSystematicCovMatrix(SingleOutput& sys_covmat) {
  t_["cov"].input(sys_covmat);
}

Segment CovariatedPrediction::resolveSegment(Atypes args,
                                             const Segment& iseg) {
  Segment ret{0, 0};
  for (size_t i = 0; i < iseg.i; ++i) {
    ret.i += args[i].size();
  }
  for (size_t i = iseg.i; i < iseg.i+iseg.n; ++i) {
    ret.n += args[i].size();
  }
  return ret;
}

void CovariatedPrediction::resolveCovarianceActions(Atypes args) {
  for (CovarianceAction &act: m_covactions) {
    if (act.a) {
      act.x = resolveSegment(args, *act.a);
    }
    if (act.b) {
      act.y = resolveSegment(args, *act.b);
    }
  }
}

void CovariatedPrediction::calculateTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0];
  m_size=0u;
  if (!m_finalized and !m_prediction_ready) {
    throw args.undefined();
  }
  if (args.size() == 0) {
    throw fargs.rets.error(ret);
  } else if (args.size() == 1) {
    ret = args[0];
  } else {
    size_t size = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      size += args[i].size();
    }
    ret = DataType().points().shape(size);
  }
  resolveCovarianceActions(args);
  m_size=ret.size();
}

/**
   * @brief calculates theoretical prediction. Implementation detail -- just
   * copies data from inputs to outputs
*/

void CovariatedPrediction::calculatePrediction(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto* buf = fargs.rets[0].x.data();
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    buf = std::copy(arg.x.data(), arg.x.data()+arg.type.size(), buf);
  }
}

void CovariatedPrediction::calculateCovbaseTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  if (!m_finalized) {
    throw args.undefined();
  }
  m_diagonal_covbase=true;
  for (size_t i = 0; i < args.size(); ++i) {
    const CovarianceAction& act = m_covactions[i];
    std::vector<size_t> expected;
    if (act.x) {
      expected.push_back(act.x->n);
    }
    if (act.y && act.y->n != 1) {
      expected.push_back(act.y->n);
    }
    std::vector<size_t> argshape = args[i].shape;
    while (argshape.size() > 1 && argshape.back() == 1) {
      argshape.pop_back();
    }

    if(act.action==CovarianceAction::Diagonal && argshape.size()>1 && expected.size()==1){
      // In case action is diagonal, accept both:
      //   1d - fill diagonal
      //   2d - fill block
      expected.push_back(expected.front());
    }

    if(act.action==CovarianceAction::Block || argshape.size()>1){
      m_diagonal_covbase=false;
    }

    if (argshape != expected) {
      std::string s = "invalid block size (";
      for (size_t j = 0; j < argshape.size(); ++j) {
        s += boost::lexical_cast<std::string>(argshape[j]);
        s += j != argshape.size()-1 ? ", " : ")";
      }
      s += ", expected (";
      for (size_t j = 0; j < expected.size(); ++j) {
        s += boost::lexical_cast<std::string>(expected[j]);
        s += j != expected.size()-1 ? ", " : ")";
      }
      throw args.error(args[i], s);
    }
  }

  auto& ret=rets[0];
  if(m_diagonal_covbase){
    ret.points().shape(size());
  }
  else{
    ret.points().shape(size(), size());
  }
}

void CovariatedPrediction::calculateCovbase(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0];
  ret.x.setZero();
  for (size_t i = 0; i < args.size(); ++i) {
    auto& arg = args[i];
    const CovarianceAction &act = m_covactions[i];
    switch (act.action) {
      case CovarianceAction::Diagonal:
        if(arg.type.shape.size()>1){
          ret.mat.block(act.x->i, act.x->i, act.x->n, act.x->n) = arg.arr2d;
        }
        else{
          if(m_diagonal_covbase){
            ret.arr.segment(act.x->i, act.x->n) = arg.arr;
          }
          else{
            ret.mat.diagonal().segment(act.x->i, act.x->n) = arg.arr;
          }
        }
        break;
      case CovarianceAction::Block:
        /* std::cout << arg.arr; */
        ret.mat.block(act.x->i, act.y->i, act.x->n, act.y->n) = arg.arr2d;
        ret.mat.block(act.y->i, act.x->i, act.y->n, act.x->n) = arg.arr2d.transpose();
        break;
    }
  }
}

/**
   * @brief Checks that dimensions of vectors and dimension of covbase
   * matrix match and preallocate covmatrix storage.
*/
void CovariatedPrediction::calculateCovTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0];

  size_t req_size=size();
  m_diagonal_cov=true;
  for (size_t i = 0; i < args.size(); ++i) {
    auto& argi=args[i];

    if(argi.shape.size()>1){
      m_diagonal_cov=false;
    }
    for(auto dim: argi.shape){
      if(dim!=req_size){
        throw args.error(argi, "argument has invalid size");
      }
    }
  }

  if(m_diagonal_cov){
    ret.points().shape(req_size);
  }
  else{
    ret.points().shape(req_size, req_size);

    m_llt = LLT(size());
    using dataPtrMatrix_t = decltype(m_llt.matrixRef().data());
    ret.preallocated(const_cast<dataPtrMatrix_t>(m_llt.matrixRef().data()));
  }
}

/**
   * @brief Update covbase matrix using rank-1 updates with vectors of
   * derivatives
*/
void CovariatedPrediction::calculateCov(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0];

  Eigen::MatrixXd fullcovmat;
  if(m_diagonal_cov){
    fullcovmat.resize(ret.arr.size(), 1);
  }
  else{
    fullcovmat.resize(ret.mat.rows(), ret.mat.cols());
  }
  fullcovmat.setZero();

  for (size_t i = 0; i < args.size(); ++i) {
    auto& arg=args[i];
    bool argdiag = arg.type.shape.size()==1;

    if(argdiag==m_diagonal_cov){
        if (argdiag) {
          fullcovmat.array()+=arg.arr;
          }
        else {
          fullcovmat.matrix()+=arg.mat;
        }
    }
    else{
      fullcovmat+=arg.vec.asDiagonal();
    }
  }

  if(m_diagonal_cov){
    ret.arr=fullcovmat.col(0).array().sqrt();
  }
  else{
    m_llt.compute(fullcovmat);
  }
}

/**
   * @brief Returns size of theoretical prediction (sum of sizes of all
   * prediction inputs.)
*/
size_t CovariatedPrediction::size() const {
  return m_size;
}

/**
   * @brief Force update of theoretical prediction.
*/
void CovariatedPrediction::update() const {
  transformations[0].touch();
}
