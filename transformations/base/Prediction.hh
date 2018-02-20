#pragma once

#include "GNAObject.hh"

/**
 * @brief Concatenation transformations. Concatenates several predictions in a single array.
 * @author Dmitry Taychenachev
 * @date 2015
 */
class Prediction: public GNASingleObject,
                  public TransformationBind<Prediction> {
public:
  Prediction();                                     ///< Default constructor.
  Prediction(const Prediction &other);              ///< Copy constructor.

  Prediction &operator=(const Prediction &other);   ///< Copy assignment.

  InputDescriptor append(const char* name);         ///< Add a named input.
  InputDescriptor append(SingleOutput &out);        ///< Add an input and connect an output to it.

  void calculateTypes(Atypes args, Rtypes rets);    ///< MemTypesFunction.
  void calculatePrediction(Args args, Rets rets);   ///< MemFunction.

  size_t size() const;                              ///< The size of the output.
  void update() const;                              ///< Force update of the calculation.
protected:
  Handle m_transform;                               ///< The transformation instance.
};
