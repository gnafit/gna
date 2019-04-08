#pragma once

#include <string>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/noncopyable.hpp>

#include "ParametrizedEntry.hh"
#include "ParametrizedEntry.hh"
#include "dependant.hh"

namespace ParametrizedTypes {
    using view_clone_allocator = boost::view_clone_allocator;
    using SourcesContainer = boost::ptr_vector<ParametrizedEntry, view_clone_allocator>;

    class ParametrizedBase;
    class EvaluableEntry: public boost::noncopyable {
    public:
        EvaluableEntry(std::string name, const SourcesContainer &sources,
                       dependant<void> dependant, const ParametrizedBase *parent);
        EvaluableEntry(const EvaluableEntry &other, const ParametrizedBase *parent);

        std::string name;
        SourcesContainer sources;
        dependant<void> dep;
        const ParametrizedBase *parent;
    };
}
