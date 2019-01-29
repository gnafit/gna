#pragma once

#include <memory>
#include "Args.hh"
#include "Rets.hh"
#include "Ints.hh"
#include "Atypes.hh"
#include "Rtypes.hh"
#include "Itypes.hh"

namespace TransformationTypes
{
  template<typename FloatType,typename SizeType> struct GPUFunctionArgsT;
  template<typename SourceType,typename SinkType> struct EntryT;

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
  template<typename SourceFloatType, typename SinkFloatType>
  struct FunctionArgsT {
    using EntryType              = EntryT<SourceFloatType,SinkFloatType>;
    using FunctionArgsType       = FunctionArgsT<SourceFloatType,SinkFloatType>;
    using GPUFunctionArgsType    = GPUFunctionArgsT<SourceFloatType, unsigned int>;
    using GPUFunctionArgsPtr     = std::unique_ptr<GPUFunctionArgsType>;

    FunctionArgsT(EntryType* e) : args(e), rets(e), ints(e), m_entry(e) {  }         ///< Constructor.
    FunctionArgsT(const FunctionArgsType& other) : FunctionArgsT(other.m_entry) {  } ///< Copy constructor.
    ~FunctionArgsT();

    ArgsT<SourceFloatType,SinkFloatType> args; ///< arguments, or transformation inputs (read-only)
    RetsT<SourceFloatType,SinkFloatType> rets; ///< return values, or transformation outputs (writable)
    IntsT<SourceFloatType,SinkFloatType> ints; ///< preallocated data arrays for the transformation's internal usage (writable)

    GPUFunctionArgsPtr gpu; ///< GPU function arguments

    void requireGPU();                    ///< Initialize GPU function arguments
    void updateTypes();                   ///< Update arguments and types

    private:
      EntryType *m_entry; ///< Entry instance to access Sinks.
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
  template<typename SourceFloatType, typename SinkFloatType>
  struct TypesFunctionArgsT {
    using EntryType = EntryT<SourceFloatType,SinkFloatType>;
    TypesFunctionArgsT(EntryType* e) : args(e), rets(e), ints(e) {  } ///< Constructor.

    AtypesT<SourceFloatType,SinkFloatType> args; ///< arguments'/inputs' data types (read-only)
    RtypesT<SourceFloatType,SinkFloatType> rets; ///< return values'/outputs' data  types (writable)
    ItypesT<SourceFloatType,SinkFloatType> ints; ///< preallocated storage's data types (writable)
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
  template<typename SourceFloatType, typename SinkFloatType>
  struct StorageTypesFunctionArgsT {
    using TypesFunctionArgsType = TypesFunctionArgsT<SourceFloatType,SinkFloatType>;
    StorageTypesFunctionArgsT(TypesFunctionArgsType& fargs) : args(fargs.args), rets(fargs.rets), ints(fargs.ints) {  } ///< Constructor.

    AtypesT<SourceFloatType,SinkFloatType>& args;       ///< arguments'/inputs' data types (read-only)
    const RtypesT<SourceFloatType,SinkFloatType>& rets; ///< return values'/outputs' data  types (read-only)
    ItypesT<SourceFloatType,SinkFloatType>& ints;       ///< preallocated storage's data types (writable)
  };

  using FunctionArgs = FunctionArgsT<double,double>;
  using TypesFunctionArgs = TypesFunctionArgsT<double,double>;
  using StorageTypesFunctionArgs = StorageTypesFunctionArgsT<double,double>;
}

