#include "PredictionSet.hh"

PredictionSet::PredictionSet() {
  transformation_("prediction")
    .output("prediction", DataType().points().any())
    .types([](Atypes args, Rtypes rets) {
        if (args.size() > 0) {
          size_t size = 0;
          for (size_t i = 0; i < args.size(); ++i) {
            size += args[i].size;
          }
          rets[0] = DataType().points().size(size);
          return Status::Success;
        } else {
          return Status::Undefined;
        }
      })
    .func([](Args args, Rets rets) {
        size_t nargs = args.size();
        double *buf = rets[0].x.data();
        for (size_t i = 0; i < nargs; ++i) {
          auto &arg = args[i];
          buf = std::copy(arg.x.data(), arg.x.data()+arg.x.size(), buf);
        }
        return Status::Success;
      });
  m_transform = t_["prediction"];
}

PredictionSet::PredictionSet(const PredictionSet &/* other */)
  : m_transform(t_["prediction"])
{
}

PredictionSet &PredictionSet::operator=(const PredictionSet &/* other */) {
  m_transform = t_["prediction"];
  return *this;
}

void PredictionSet::add(const OutputDescriptor &out) {
  t_["prediction"].input(out.channel()).connect(out);
}

size_t PredictionSet::size() const {
  return m_transform[0].type.size;
}

void PredictionSet::update() const {
  m_transform.update(0);
}

const double *PredictionSet::data() const {
  return m_transform[0].x.data();
}
