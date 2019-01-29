#pragma once

#include "EntryHandle.hh"
#include "OutputHandle.hh"

template<typename SourceFloatType, typename SinkFloatType> struct EntryT;
template<typename FloatType> class OutputHandleT;

/**
 * @brief User-end wrapper for the Entry class that gives user an access to the actual Entry.
 *
 * The class is used for the dependency tree plotting via graphviz module.
 *
 * @author Maxim Gonchar
 * @date 12.2017
 */
class OpenHandle : public TransformationTypes::Handle {
public:
    using Entry= EntryT<double,double>;
    OpenHandle(const TransformationTypes::Handle& other) : TransformationTypes::Handle(other){}; ///< Constructor. @param other -- Handle instance.
    TransformationTypes::Entry* getEntry() { return m_entry; }                               ///< Get the Entry pointer.
};

class OpenOutputHandle : public TransformationTypes::OutputHandleT<double> {
public:
    using Entry= EntryT<double,double>;
    using OutputHandle = OutputHandleT<double>;
    OpenOutputHandle(const OutputHandle& other) : OutputHandle(other){}; ///< Constructor. @param other -- OutputHandle instance.
    TransformationTypes::Entry* getEntry() { return m_sink->entry; }     ///< Get thy Entry pointer.
};
