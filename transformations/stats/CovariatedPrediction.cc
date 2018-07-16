#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include "CovariatedPrediction.hh"

typedef CovariatedPrediction::Segment Segment;

CovariatedPrediction::CovariatedPrediction()
  : m_finalized(false)
{
  transformation_("prediction")
    .output("prediction")
    .types(&CovariatedPrediction::calculateTypes)
    .func(&CovariatedPrediction::calculatePrediction)
    ;
  transformation_("covbase")
    .output("L")
    .types(&CovariatedPrediction::calculateCovbaseTypes)
    .func(&CovariatedPrediction::calculateCovbase)
    ;
  transformation_("cov")
    .input("Lbase")
    .output("L")
    .types(&CovariatedPrediction::calculateCovTypes)
    .func(&CovariatedPrediction::calculateCov)
    ;
  m_transform = t_["prediction"];
}

CovariatedPrediction::CovariatedPrediction(const CovariatedPrediction &other)
  : m_transform(t_["prediction"]),
    m_inputs(other.m_inputs), m_finalized(other.m_finalized), m_prediction_ready(other.m_prediction_ready)
{
}

CovariatedPrediction& CovariatedPrediction::operator=(const CovariatedPrediction &other) {
  m_transform = t_["prediction"];
  m_inputs = other.m_inputs;
  m_finalized = other.m_finalized;
  m_prediction_ready = other.m_prediction_ready;
  return *this;
}

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
  auto& rets=fargs.rets;
  if (!m_finalized and !m_prediction_ready) {
    throw args.undefined();
  }
  if (args.size() == 0) {
    throw rets.error(rets[0]);
  } else if (args.size() == 1) {
    rets[0] = args[0];
  } else {
    size_t size = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      size += args[i].size();
    }
    rets[0] = DataType().points().shape(size);
  }
  resolveCovarianceActions(args);
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
  rets[0] = DataType().points().shape(size(), size());
  /* m_lltbase = LLT(size()); */
  /* rets[0].preallocated(m_lltbase.matrixRef().data()); */
  m_covbase.resize(size(), size());
  rets[0].preallocated(m_covbase.matrix().data());
}

void CovariatedPrediction::calculateCovbase(FunctionArgs fargs) {
  auto& args=fargs.args;
  m_covbase.setZero();
  for (size_t i = 0; i < args.size(); ++i) {
    const CovarianceAction &act = m_covactions[i];
    switch (act.action) {
    case CovarianceAction::Diagonal:
      m_covbase.matrix().diagonal().segment(act.x->i, act.x->n) = args[i].arr;
      break;
    case CovarianceAction::Block:
      std::cout << args[i].arr;
      m_covbase.block(act.x->i, act.y->i, act.x->n, act.y->n) = args[i].arr;
      break;
    }
  }

  /* Force materialization of matrix */
  (void)fargs.rets[0].mat;
  /* m_lltbase.compute(m_covbase); */
}

/**
   * @brief Checks that dimensions of vectors and dimension of covbase
   * matrix match and preallocate covmatrix storage.
*/
void CovariatedPrediction::calculateCovTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  /* for (size_t i = 1; i < args.size(); ++i) {
   *   if (args[i].size() != size()) {
   *     throw args.error(args[i], "rank1 vec size does not match prediction size");
   *   }
   * } */
  rets[0] = args[0];
  m_llt = LLT(size());
  using dataPtrMatrix_t = decltype(m_llt.matrixRef().data());
  rets[0].preallocated( const_cast<dataPtrMatrix_t>(m_llt.matrixRef().data()) );
}

/**
   * @brief Update covbase matrix using rank-1 updates with vectors of
   * derivatives
*/
void CovariatedPrediction::calculateCov(FunctionArgs fargs) {
  auto& args=fargs.args;
  (void)args[0].mat;
  auto full_covmat = m_covbase;
  if (args.size() > 1) {
      auto& sys_covmat = args[1].arr2d;
      full_covmat += sys_covmat;
  }
  m_llt.compute(full_covmat);
  /* for (size_t i = 1; i < args.size(); ++i) {
   *   m_llt.rankUpdate(args[i].vec);
   * } */
  (void)fargs.rets[0].mat;
}

/**
   * @brief Returns size of theoretical prediction (sum of sizes of all
   * prediction inputs.)
*/
size_t CovariatedPrediction::size() const {
  return m_transform[0].type.size();
}

/**
   * @brief Force update of theoretical prediction.
*/
void CovariatedPrediction::update() const {
  m_transform.update(0);
}
