#pragma once

#include "Data.hh"
#include "OutputHandle.hh"

/**
 * @brief The object with single output.
 *
 * Base class declares the SingleOutput::single() function to be used for object with single output.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class SingleOutput {
public:
  virtual ~SingleOutput() { }                                  ///< Destructor.
  virtual TransformationTypes::OutputHandle single() = 0;      ///< Return the single output.
  const double *data() { return single().data(); }             ///< Return the single output's Data. The relevant Entry is updated if needed.
  //const double *view() { return single().view(); }           ///< Return the view on a single output's Data.
  const DataType &datatype() { return single().datatype(); }   ///< Return the single output's DataType.
};

