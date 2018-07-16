#include "Prediction.hh"

/**
 * @brief Default constructor.
 *
 * Outputs:
 *   `prediction.prediction` -- the concatenated arrays.
 */
Prediction::Prediction() {
  transformation_("prediction")
    .output("prediction")
    .types(&Prediction::calculateTypes)
    .func(&Prediction::calculatePrediction)
    ;
  m_transform = t_["prediction"];
}

/**
 * @brief Copy constructor.
 * Copies the transformation from the other instance of Prediction.
 * @param other -- the other Prediction instance.
 */
Prediction::Prediction(const Prediction &other)
  : m_transform(t_["prediction"])
{
}

/**
 * @brief Copy assignment.
 * Copies the transformation from the other instance of Prediction.
 * @param other -- the other Prediction instance.
 */
Prediction &Prediction::operator=(const Prediction &other) {
  m_transform = t_["prediction"];
  return *this;
}

/**
 * @brief Create the new input by name and leave it not connected.
 * @param name -- new input name.
 * @return InputDescriptor for newly created input.
 */
InputDescriptor Prediction::append(const char* name) {
  return InputDescriptor(t_["prediction"].input(name));
}

/**
 * @brief Create the new input and connect an output to it.
 * Defines a name of the input by output's name.
 * @param obs -- SingleOutput instance.
 * @return InputDescriptor for newly created input.
 */
InputDescriptor Prediction::append(SingleOutput &obs) {
  return InputDescriptor(t_["prediction"].input(obs));
}

/**
 * @brief MemTypesFunction.
 * Sums the sizes of the input arrays/histograms.
 * @exception SinkTypeError in case there are no inputs.
 */
void Prediction::calculateTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
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

/**
 * @brief MemFunction.
 * Copies the data from each input into concatenated output.
 */
void Prediction::calculatePrediction(FunctionArgs fargs) {
  auto& args=fargs.args;
  double *buf=fargs.rets[0].x.data();

  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    /* std::copy returns an iterator to the last inserted element  */
    buf = std::copy(arg.x.data(), arg.x.data()+arg.type.size(), buf);
  }
}

/**
 * @brief The size of the output.
 * @return the size of the output.
 */
size_t Prediction::size() const {
  return m_transform[0].type.size();
}

/**
 * @brief Force update of the calculation.
 */
void Prediction::update() const {
  m_transform.update(0);
}
