#pragma once

#include <list>
#include <tuple>

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
class TransformationBind {
public:
  TransformationBind() { }                                                   ///< Default constructor.
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
   * @param obj -- the pointer to the TransformationBind.
   * @param name -- the transformation name.
   * @return transformation Initializer.
   */
  TransformationTypes::Initializer<Derived> transformation_(TransformationTypes::Base *base, const std::string &name) {
    return TransformationTypes::Initializer<Derived>(base, this, name);
  }

private:
  friend class TransformationTypes::Initializer<Derived>;
  typedef typename TransformationTypes::Initializer<Derived> Initializer;
  typedef typename Initializer::MemFunction MemFunction;
  typedef typename Initializer::MemTypesFunction MemTypesFunction;

  /**
   * @brief Return `this` casted to the Derived type (CRTP).
   * @return `this`.
   */
  Derived *obj() { return static_cast<Derived*>(this); }

  /**
   * @copydoc TransformationBind::obj()
   */
  const Derived *obj() const { return static_cast<const Derived*>(this); }

  /**
   * @brief Return `this` cast to the TransformationTypes::Base.
   * @return `this`.
   */
  TransformationTypes::Base *baseobj() {
    return static_cast<TransformationTypes::Base*>(obj());
  }

  /**
   * @copydoc TransformationBind::baseobj()
   */
  const TransformationTypes::Base *baseobj() const {
    return static_cast<const TransformationTypes::Base*>(obj());
  }

  std::list<std::tuple<size_t, MemFunction>> m_memFuncs;                      ///< List with MemFunction objects arranged correspondingly to each Entry from Base.
  std::list<std::tuple<size_t, size_t, MemTypesFunction>> m_memTypesFuncs;    ///< List with MemTypesFunction objects arranged correspondingly to each Entry from Base.

  /**
   * @brief Add new MemFunction.
   * @param idx -- Entry index.
   * @param func -- the function.
   */
  void addMemFunction(size_t idx, MemFunction func) {
    m_memFuncs.emplace_back(idx, func);
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
   * @brief Bind each of the MemFunction and MemTypesFunction objects to `this` of the transformation.
   * The method replaces the relevant Function and TypesFunction of the Entry by the binded MemFunction and MemTypesFunction.
   */
  void rebindMemFunctions() {
    using namespace std::placeholders;
    auto &entries = baseobj()->m_entries;
    for (const auto &f: m_memFuncs) {
      auto idx = std::get<0>(f);
      entries[idx].fun = std::bind(std::get<1>(f), obj(), _1, _2);
    }
    for (const auto &f: m_memTypesFuncs) {
      auto idx = std::get<0>(f);
      auto fidx = std::get<1>(f);
      entries[idx].typefuns[fidx] = std::bind(std::get<2>(f), obj(), _1, _2);
    }
  }
}; /* class TransformationBind */

