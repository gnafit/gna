#pragma once

#include "Random.hh"
#include "GNAObject.hh"
#include "GNAObjectBindkN.hh"

class NormalToyMC: public GNAObjectBindkN,
                   public TransformationBind<NormalToyMC> {
public:
  NormalToyMC(bool autofreeze=true);

  void add(SingleOutput& theory, SingleOutput &cov) { add_inputs(SingleOutputsContainer({&theory, &cov}));  }
  void nextSample();

  void reset() { m_distr.reset(); }

  TransformationDescriptor add_transformation(const std::string& name="");
protected:
  void calcTypes(TypesFunctionArgs fargs);
  void calcToyMC(FunctionArgs fargs);

  std::normal_distribution<> m_distr;

  bool m_autofreeze;
};
