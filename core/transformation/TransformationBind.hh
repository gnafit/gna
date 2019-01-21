#pragma once

#include <list>
#include <tuple>
#include <functional>

#include "TransformationBase.hh"
#include "Initializer.hh"

/**
 * @brief Base class for the transformation definition.
 *
 * Each GNA transformation class is defined by deriving to base classes:
 *   - GNAObject or GNASingleObject (deriving from TransformationTypes::Base and ParametrizedTypes::Base).
 *   - TransformationBind<class>.
 *
 * TransformationBind class does the bookkeeping for MemFunction and MemTypesFunction. By defining CRTP
 * TransformationBind::obj() method it facilitates the binding of the first
 * argument of MemFunction and MemTypesFunction objects to `this` of the transformation.
 *
 * @tparam Derived -- derived class type. See CRTP concept.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
template <typename Derived>
class TransformationBind: public virtual TransformationTypes::Base {
public:
  TransformationBind() = default;                                                   ///< Default constructor.
  /**
   * @brief Clone constructor.
   *
   * The constructor copies the list of MemFunction objects and rebinds them to `this`.
   *
   * @param other -- TransformationBind to copy MemFunction and MemTypesFunction objects from.
   */
  TransformationBind(const TransformationBind<Derived> &other)
    : m_memFuncs(other.m_memFuncs), m_memTypesFuncs(other.m_memTypesFuncs)
  {
    rebindMemFunctions();
  }

  /**
   * @brief Clone assignment. Works the same was as clone constructor.
   * @copydoc TransformationBind::TransformationBind(const TransformationBind<Derived>&)
   */
  TransformationBind<Derived> &operator=(const TransformationBind<Derived> &other) {
    m_memFuncs = other.m_memFuncs;
    m_memTypesFuncs = other.m_memTypeFuncs;
    rebindMemFunctions();
    return *this;
  }

  /**
   * @brief Initialize the new transformation Entry.
   * @param name -- the transformation name.
   * @return transformation Initializer.
   */
  TransformationTypes::Initializer<Derived> transformation_(const std::string &name) {
    return TransformationTypes::Initializer<Derived>(this, name);
  }
protected:
  friend class TransformationTypes::Initializer<Derived>;
  using Initializer = typename TransformationTypes::Initializer<Derived>;
  using MemFunction = typename Initializer::MemFunction;
  using MemTypesFunction = typename Initializer::MemTypesFunction;
  using MemStorageTypesFunction = typename Initializer::MemStorageTypesFunction;

private:
  using MemTypesFunctionGMap = std::list<std::tuple<size_t, size_t, MemTypesFunction>>;
  using MemStorageTypesFunctionGMap = std::list<std::tuple<size_t, std::string, size_t, MemStorageTypesFunction>>;

  /**
   * @brief Return `this` casted to the Derived type (CRTP).
   * @return `this`.
   */
  Derived *obj() { return static_cast<Derived*>(this); }

  /**
   * @copydoc TransformationBind::obj()
   */
  const Derived *obj() const { return static_cast<const Derived*>(this); }

  std::list<std::tuple<size_t, std::string, MemFunction>> m_memFuncs;  ///< List with MemFunction objects arranged correspondingly to each Entry from Base.
  MemTypesFunctionGMap m_memTypesFuncs;                                ///< List with MemTypesFunction objects arranged correspondingly to each Entry from Base.
  MemStorageTypesFunctionGMap m_memStorageFuncs;

  /**
   * @brief Add new MemFunction.
   * @param idx -- Entry index.
   * @param name -- the name of a function.
   * @param func -- the function.
   */
  void addMemFunction(size_t idx, const std::string& name, MemFunction func) {
    m_memFuncs.emplace_back(idx, name, func);
  }

  /**
   * @brief Add new MemTypesFunction.
   * @param idx -- Entry index.
   * @param fidx -- function index (Each Entry may have several TypeFunction objects).
   * @param func -- the function.
   */
  void addMemTypesFunction(size_t idx, size_t fidx, MemTypesFunction func) {
    m_memTypesFuncs.emplace_back(idx, fidx, func);
  }

  /**
   * @brief Add new MemStorageTypesFunction that will initialize the storage for a particular named Function.
   * @param idx -- Entry index.
   * @param fname -- Function name.
   * @param fidx -- StorageTypesFunction index (Each Entry may have several StorageTypeFunction objects).
   * @param func -- the StorageTypesFunction.
   */
  void addMemStorageTypesFunction(size_t idx, const std::string& fname, size_t fidx, MemStorageTypesFunction func) {
    m_memStorageFuncs.emplace_back(idx, fname, fidx, func);
  }

  /**
   * @brief Bind MemFunction/MemTypesFunction/MemStorageTypesFunction to a particular class instance
   * @param function -- MemFunction, MemTypesFunction or MemStorageTypesFunction
   * @return Function, TypesFunction or StorageTypesFunction bound to this, respectively
   */
  template<class FunctionArgsType>
  std::function<void(FunctionArgsType&)> bind(std::function<void(Derived*, FunctionArgsType&)> func){
    auto* ptr=this->obj();
    return [func,ptr](FunctionArgsType& fargs){
      func(ptr, fargs);
    };
  }

  /**
   * @brief Bind each of the MemFunction and MemTypesFunction objects to `this` of the transformation.
   * The method replaces the relevant Function and TypesFunction of the Entry by the binded MemFunction and MemTypesFunction.
   *
   * @todo Should be tested.
   */
  void rebindMemFunctions() {
    for (const auto &f: m_memFuncs) {
      auto  idx   = std::get<0>(f);
      auto& name  = std::get<1>(f);
      auto& mfunc = std::get<2>(f);
      auto& entry = m_entries[idx];
      auto& func  = this->template bind<>(mfunc);
      entry.functions.at(name).fun = func;
      if(entry.funcname==name) {
        entry.fun = func;
      }
    }
    for (const auto &f: m_memTypesFuncs) {
      auto  idx = std::get<0>(f);
      auto  fidx = std::get<1>(f);
      auto& func = std::get<2>(f);
      m_entries[idx].typefuns[fidx] = this->template bind<>(func);
    }
    for (const auto &f: m_memStorageFuncs) {
      auto  idx  = std::get<0>(f);
      auto& name = std::get<1>(f);
      auto  fidx = std::get<2>(f);
      auto& func = std::get<3>(f);
      m_entries[idx].functions[name].typefuns[fidx] = this->template bind<>(func);
    }
  }
}; /* class TransformationBind */

