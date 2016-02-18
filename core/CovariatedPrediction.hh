#ifndef COVARIATEDPREDICTION_H
#define COVARIATEDPREDICTION_H

#include <boost/optional.hpp>

#include "GNAObject.hh"
#include "UncertainParameter.hh"
#include "Covariance.hh"

class CovariatedPrediction: public GNAObject,
                            public Transformation<CovariatedPrediction> {
public:
  struct Segment {
    size_t i, n;
  };

  CovariatedPrediction();
  CovariatedPrediction(const CovariatedPrediction &other);

  CovariatedPrediction &operator=(const CovariatedPrediction &other);

  void append(SingleOutput &obs);
  void finalize();

  size_t blockOffset(OutputDescriptor inp);
  size_t blocksCount() const;
  void covariate(SingleOutput &cov,
                 SingleOutput &obs1, size_t n1,
                 SingleOutput &obs2, size_t n2);
  void rank1(SingleOutput &vec);

  Segment resolveSegment(Atypes args, const Segment &iseg);
  void resolveCovarianceActions(Atypes args);
  void calculateTypes(Atypes args, Rtypes rets);
  void calculatePrediction(Args args, Rets rets);
  void calculateCovbaseTypes(Atypes args, Rtypes rets);
  void calculateCovbase(Args args, Rets rets);
  void calculateCovTypes(Atypes args, Rtypes rets);
  void calculateCov(Args args, Rets rets);

  size_t size() const;

  void update() const;
protected:
  struct CovarianceAction {
    enum Action {
      Diagonal, Block
    };

    CovarianceAction(Action act) : action(act) { }

    Action action;

    boost::optional<Segment> a, b;

    bool resolved = false;
    boost::optional<Segment> x, y;
  };

  class LLT: public Eigen::LLT<Eigen::MatrixXd> {
  public:
    LLT(): Eigen::LLT<Eigen::MatrixXd>() { }
    LLT(size_t size): Eigen::LLT<Eigen::MatrixXd>(size) { }
    Eigen::MatrixXd &matrixRef() { return this->m_matrix; }
  };

  Handle m_transform;

  const Covariance *m_covariance;
  std::vector<OutputDescriptor> m_inputs;

  std::vector<CovarianceAction> m_covactions;

  bool m_finalized;
  LLT m_lltbase, m_llt;
  Eigen::ArrayXXd m_covbase;
};

#endif // COVARIATEDPREDICTION_H
