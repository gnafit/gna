#pragma once

#include "GNAObject.hh"

/**
 * @brief Concatenation transformations. Concatenates several concats in a single array.
 *
 * Outputs:
 *   `concat.concat` -- the concatenated arrays.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class Concat: public GNASingleObject,
                  public TransformationBind<Concat> {
public:
  Concat();                                     ///< Default constructor.
  Concat(const Concat &other);                  ///< Copy constructor.

  Concat &operator=(const Concat &other);       ///< Copy assignment.

  InputDescriptor append(const char* name) { return add_input(name); } ///< Add a named input.
  InputDescriptor append(SingleOutput &out){ return add_input(out); }  ///< Add an input and connect an output to it.

  InputDescriptor add_input(const char* name);  ///< Add a named input.
  InputDescriptor add_input(SingleOutput &out); ///< Add an input and connect an output to it.

  void calculateTypes(TypesFunctionArgs& fargs); ///< MemTypesFunction.
  void calculateConcat(FunctionArgs& fargs);     ///< MemFunction.

  size_t size() const;                           ///< The size of the output.
  void update() const;                           ///< Force update of the calculation.
protected:
  Handle m_transform;                            ///< The transformation instance.
};
