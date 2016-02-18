#include "Prediction.hh"

Prediction::Prediction() {
  transformation_(this, "prediction")
    .output("prediction")
    .types(&Prediction::calculateTypes)
    .func(&Prediction::calculatePrediction)
    ;
  m_transform = t_["prediction"];
}

Prediction::Prediction(const Prediction &other)
  : m_transform(t_["prediction"])
{
}

Prediction &Prediction::operator=(const Prediction &other) {
  m_transform = t_["prediction"];
  return *this;
}

void Prediction::append(SingleOutput &obs) {
  t_["prediction"].input(obs);
}

void Prediction::calculateTypes(Atypes args, Rtypes rets) {
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
}

void Prediction::calculatePrediction(Args args, Rets rets) {
  double *buf = rets[0].x.data();

  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    buf = std::copy(arg.x.data(), arg.x.data()+arg.type.size(), buf);
  }
}

size_t Prediction::size() const {
  return m_transform[0].type.size();
}

void Prediction::update() const {
  m_transform.update(0);
}
