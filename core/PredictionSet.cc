#include "PredictionSet.hh"

PredictionSet::PredictionSet() {
  transformation_("prediction",
                  {},
                  {{"prediction", DataType().series().any()}},
                  &PredictionSet::build,
                  &PredictionSet::buildTypes);
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
  out.connect(input_("prediction", out.channel()));
}

Status PredictionSet::buildTypes(ArgumentTypes args, ReturnTypes rets) {
  if (args.size() > 1) {
    size_t size = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      size += args[i].shape[0];
    }
    rets[0] = DataType().series().shape({(int)size});
  } else if (args.size() > 0) {
    rets[0] = args[0];
  } else {
    return Status::Undefined;
  }
  return Status::Success;
}

Status PredictionSet::build(Arguments args, Returns rets) {
  size_t nargs = args.size();
  double *buf = rets[0].buffer;
  for (size_t i = 0; i < nargs; ++i) {
    auto &arg = args[i];
    buf = std::copy(arg.buffer, arg.buffer+arg.size, buf);
  }
  return Status::Success;
}

size_t PredictionSet::size() const {
  return m_transform[0].size;
}

void PredictionSet::update() const {
  m_transform.update(0);
}

const double *PredictionSet::data() const {
  return m_transform[0].buffer;
}

const Data<const double> &PredictionSet::view() const {
  return m_transform[0].view();
}
