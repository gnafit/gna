#pragma once

#include <boost/optional.hpp>

#include "GNAObject.hh"
#include "UncertainParameter.hh"

class CovariatedPrediction: public GNAObject,
                            public TransformationBind<CovariatedPrediction> {
public:

    /**
   * @brief Defines a segment in covariance
   */
  struct Segment {
    size_t i, n;

    void dump() const {
      printf("i=%zu, n=%zu, i+n=%zu", i, n, i+n);
    }
  };

  CovariatedPrediction();
  CovariatedPrediction(const CovariatedPrediction &other);

  CovariatedPrediction &operator=(const CovariatedPrediction &other);

  void append(SingleOutput &obs);
  void finalize();
  void prediction_ready();

  size_t blockOffset(OutputDescriptor inp);
  size_t blocksCount() const noexcept;
  void covariate(SingleOutput &cov,
                 SingleOutput &obs1, size_t n1,
                 SingleOutput &obs2, size_t n2);
  void rank1(SingleOutput &vec);

  Segment resolveSegment(Atypes args, const Segment &iseg);
  void resolveCovarianceActions(Atypes args);
  void calculateTypes(TypesFunctionArgs fargs);
  void calculatePrediction(FunctionArgs fargs);
  void calculateCovbaseTypes(TypesFunctionArgs fargs);
  void calculateCovbase(FunctionArgs fargs);
  void calculateCovTypes(TypesFunctionArgs fargs);
  void calculateCov(FunctionArgs fargs);
  void addSystematicCovMatrix(SingleOutput& sys_covmat);

  size_t size() const;

  void update() const;
protected:

    /**
   * @brief Defines an action to perform on a given segment
   * @param Action -- Either CovarianceAction::Diagonal or CovarianceAction::Block
   */
  struct CovarianceAction {
    enum Action {
      Diagonal, Block
    };

    explicit CovarianceAction(Action act) : action(act) { }

    Action action;

    boost::optional<Segment> a, b;

    bool resolved = false;
    boost::optional<Segment> x, y;

    void dump() const {
      printf("Covariance action on %s:", action==Diagonal ? "diag" : "block");
      if(a) { printf("a "); a->dump(); }
      if(x) { printf(", x "); x->dump(); }
      if(b) { printf(", b "); b->dump(); }
      if(y) { printf(", y "); y->dump(); }
      printf("\n");
    }
  };

/**
   * @brief Stores LLT decomposition matrix and provides access to it
   * @param size -- size of a matrix to be allocated.
*/
 /* TODO: Also would need change to play along with floats. */
  class LLT: public Eigen::LLT<Eigen::MatrixXd> {
  public:
    explicit LLT(): Eigen::LLT<Eigen::MatrixXd>() { }
    explicit LLT(size_t size): Eigen::LLT<Eigen::MatrixXd>(size) { }
    Eigen::MatrixXd& matrixRef() { return this->m_matrix; }
  };

  Handle m_transform;

  std::vector<OutputDescriptor> m_inputs;

  std::vector<CovarianceAction> m_covactions;

  bool m_finalized;
  bool m_prediction_ready;
  LLT m_llt;
};
