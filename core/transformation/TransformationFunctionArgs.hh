#pragma once

#include "Args.hh"
#include "Rets.hh"
#include "Ints.hh"
#include "Atypes.hh"
#include "Rtypes.hh"
#include "Itypes.hh"

namespace TransformationTypes
{
  struct Entry;

  /**
   * @brief Transformation Function arguments.
   *
   * FunctionArgs instance is pased to the Function and contains the necessary data:
   *   - inputs;
   *   - outputs;
   *   - internal storage.
   *
   * Inputs are available read-only.
   *
   * @author Maxim Gonchar
   * @date 07.2018
   */
  struct FunctionArgs {
    FunctionArgs(Entry* e) : args(e), rets(e), ints(e) {  } ///< Constructor.

    Args args; ///< arguments, or transformation inputs (read-only)
    Rets rets; ///< return values, or transformation outputs (writable)
    Ints ints; ///< preallocated data arrays for the transformation's internal usage (writable)
  };

  /**
   * @brief Transformation TypesFunction arguments.
   *
   * TypesFunctionArgs instance is pased to the typesFunction and contains the necessary data types:
   *   - inputs' data types;
   *   - outputs' data types;
   *   - internal storages' data types.
   *
   * The input data types are available read-only.
   *
   * @author Maxim Gonchar
   * @date 07.2018
   */
  struct TypesFunctionArgs {
    TypesFunctionArgs(Entry* e) : args(e), rets(e), ints(e) {  } ///< Constructor.

    Atypes args; ///< arguments'/inputs' data types (read-only)
    Rtypes rets; ///< return values'/outputs' data  types (writable)
    Itypes ints; ///< preallocated storage's data types (writable)
  };

  /**
   * @brief Transformation StorageTypesFunction arguments.
   *
   * TypesFunctionArgs instance is pased to the typesFunction and contains the necessary data types:
   *   - inputs' data types;
   *   - outputs' data types;
   *   - internal storages' data types.
   *
   * The input and output data types are available read-only. The StorageTypesFunction may only create
   * and modify the information about the internal storage requirements.
   *
   * @author Maxim Gonchar
   * @date 07.2018
   */
  struct StorageTypesFunctionArgs {
    StorageTypesFunctionArgs(TypesFunctionArgs& fargs) : args(fargs.args), rets(fargs.rets), ints(fargs.ints) {  } ///< Constructor.

    Atypes& args;       ///< arguments'/inputs' data types (read-only)
    const Rtypes& rets; ///< return values'/outputs' data  types (read-only)
    Itypes& ints;       ///< preallocated storage's data types (writable)
  };
}

