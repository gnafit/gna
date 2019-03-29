#pragma once

#include <iostream>
#include <string>
#include "EvaluableEntry.hh"

namespace ParametrizedTypes{
    template <typename T>
    class EvaluableHandle {
    public:
        EvaluableHandle<T>(EvaluableEntry &entry)
            : m_entry(&entry) { }
        EvaluableHandle<T>(const EvaluableHandle<T> &other)
            : EvaluableHandle<T>(*other.m_entry) { }

        const std::string &name() const { return m_entry->name; }

        size_t hash() const {return m_entry->dep.hash();}
        void dump() const;
    protected:
        EvaluableEntry *m_entry;
    };

    template <typename T>
    inline void EvaluableHandle<T>::dump() const {
        std::cerr << m_entry->name;
        std::cerr << std::endl;
    }
}
