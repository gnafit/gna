#pragma once

#include "EntryHandle.hh"
#include "OutputHandle.hh"

/**
 * @brief User-end wrapper for the Entry class that gives user an access to the actual Entry.
 *
 * The class is used for the dependency tree plotting via graphviz module.
 *
 * @author Maxim Gonchar
 * @date 12.2017
 */
template<typename SourceFloatType, typename SinkFloatType>
class OpenHandleT : public TransformationTypes::HandleT<SourceFloatType,SinkFloatType> {
public:
    using Entry =TransformationTypes::EntryT<SourceFloatType,SinkFloatType>;
    using Handle=TransformationTypes::HandleT<SourceFloatType,SinkFloatType>;
    using Handle::m_entry;
    OpenHandleT(const Handle& other) : Handle(other){};                         ///< Constructor. @param other -- Handle instance.
    Entry* getEntry() { return m_entry; }                                       ///< Get the Entry pointer.
};

template<typename SourceFloatType, typename SinkFloatType>
class OpenOutputHandleT : public TransformationTypes::OutputHandleT<SinkFloatType> {
public:
    using Entry= TransformationTypes::EntryT<SourceFloatType,SinkFloatType>;
    using OutputHandle = TransformationTypes::OutputHandleT<SinkFloatType>;
    using OutputHandle::m_sink;
    OpenOutputHandleT(const OutputHandle& other) : OutputHandle(other){}; ///< Constructor. @param other -- OutputHandle instance.
    Entry* getEntry() { return m_sink->entry; }                                           ///< Get thy Entry pointer.
};

using OpenHandle = OpenHandleT<double,double>;
using OpenOutputHandle = OpenOutputHandleT<double,double>;
