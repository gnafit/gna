#include "Prediction.hh"

Prediction::Prediction() {
  transformation_(this, "prediction")
    .output("prediction")
    .types([](Atypes args, Rtypes rets) {
        if (args.size() == 0) {
          throw rets.error(rets[0]);
        }
        if (args.size() > 0) {
          size_t size = 0;
          for (size_t i = 0; i < args.size(); ++i) {
            size += args[i].size();
          }
          rets[0] = DataType().points().shape(size);
        }
      })
    .func([](Args args, Rets rets) {
        size_t nargs = args.size();
        double *buf = rets[0].x.data();
        for (size_t i = 0; i < nargs; ++i) {
          auto &arg = args[i];
          buf = std::copy(arg.x.data(), arg.x.data()+arg.x.size(), buf);
        }
      });
  m_transform = t_["prediction"];
}

Prediction::Prediction(const Prediction &/* other */)
  : m_transform(t_["prediction"])
{
}

Prediction &Prediction::operator=(const Prediction &/* other */) {
  m_transform = t_["prediction"];
  return *this;
}

void Prediction::append(SingleOutput &data) {
  t_["prediction"].input(data);
}

size_t Prediction::size() const {
  return m_transform[0].type.size();
}

void Prediction::update() const {
  m_transform.update(0);
}

