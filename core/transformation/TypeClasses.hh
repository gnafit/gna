#pragma once

#include <boost/range/counting_range.hpp>
#include <stdexcept>
#include <vector>
#include <string>
#include <fmt/format.h>

#include "TransformationFunctionArgs.hh"

namespace TypeClasses{
    class Range {
    public:
        using RangeIterator = boost::iterator_range<boost::counting_iterator<size_t>>;

        /// End is inclusive
        Range(int begin)          : m_begin(begin), m_end(begin) {}
        Range(int begin, int end) : m_begin(begin), m_end(end) {}
        Range(const Range& other) = default;

        bool singular() const { return m_end == m_begin; }

        int getBegin() const { return m_begin; }
        int getEnd()   const { return m_end; }

        size_t getBeginAbs(size_t size, bool strict=true) const {
            size_t begin, end;
            getRangeAbs(size, begin, end, strict);
            return begin;
        }
        size_t getEndAbs(size_t size, bool strict=true)   const {
            size_t begin, end;
            getRangeAbs(size, begin, end, strict);
            return end;
        }

        RangeIterator iterate(size_t size, bool strict=true) const {
            size_t begin, end;
            getRangeAbs(size, begin, end, strict);
            if(begin==-1u){
                return boost::counting_range(begin, end);
            }
            return boost::counting_range(begin, end+1);
        }

        RangeIterator iterateSafe(size_t size) const { return iterate(size, false); }

        std::vector<size_t> vector(size_t size, bool strict=true) const {
            auto it=iterate(size, strict);
            return std::vector<size_t>(it.begin(), it.end());
        }

        void dump(){
            printf("Range %i->%i", m_begin, m_end);
            if(singular()){
                printf(" (singular)");
            }
            printf("\n");
        }
    private:
        void getRangeAbs(int size, size_t &begin, size_t &end, bool strict=true) const {
            int absbegin = m_begin>=0 ? m_begin : size+m_begin;
            int absend   = m_end>=0   ? m_end   : size+m_end;

            bool error=absend<absbegin && !((m_begin>=0) ^ (m_end>=0));
            if (absend<0 || absbegin>=size || absend<absbegin){
                begin=end=-1u;
                error|=strict;
            }
            else{
                if (absend>=size) {
                    absend=size-1;
                }
                if (absbegin<0){
                    absbegin=0;
                }
                begin=absbegin;
                end=absend;
            }

            //printf("Convert %i->%i (%i): %i->%i or %zu->%zu (error: %i)\n", m_begin, m_end, size, absbegin, absend, begin, end, (int)error);

            if(error){
                throw std::runtime_error(fmt::format("Invalid range {}->{} for size {}: {}->{}", m_begin, m_end, size, absbegin, absend));
            }
        }

        int m_begin;
        int m_end;
    };

    template<typename FloatType>
    class TypeClassT {
    public:
        using TypesFunctionArgs = TransformationTypes::TypesFunctionArgsT<FloatType,FloatType>;
        virtual void check(TypesFunctionArgs& fargs) = 0;
    };

    template<typename FloatType>
    class CheckSameTypesT : TypeClassT<FloatType> {
    private:
        using BaseClass = TypeClassT<FloatType>;
        using SelfClass = CheckSameTypesT<FloatType>;

    public:
        using TypesFunctionArgs = typename BaseClass::TypesFunctionArgs;

        CheckSameTypesT(Range argsrange, const std::string& what="") : m_argsrange(argsrange) {
            if(what=="" or what=="all"){
                m_comparison = ComparisonType::All;
            }
            else if (what=="shape"){
                m_comparison = ComparisonType::Shape;
            }
            else{
                throw std::runtime_error(fmt::format("Unknown comparison type: {}", what));
            }
        }
        CheckSameTypesT(const SelfClass& other) = default;

        void check(TypesFunctionArgs& fargs){
            auto& args = fargs.args;
            const DataType* compare_to=nullptr;
            size_t first;
            for(auto aidx: m_argsrange.iterate(args.size())){
                if(!compare_to){
                    compare_to = &args[aidx];
                    first = aidx;
                    continue;
                }
                bool eq = compare(args[aidx], *compare_to);
                if (!eq) {
                    auto msg = fmt::format("Transformation {0}: all inputs should have same {3}, {1} and {2} differ", args.name(), first, aidx, m_comparison==ComparisonType::All?"type":"shape");
                    throw args.error(args[aidx], msg);
                }
            }
        }

        bool compare(const DataType& dt1, const DataType& dt2) const {
            switch(m_comparison){
                case ComparisonType::All:
                return dt1 == dt2;
                break;

                case ComparisonType::Shape:
                return dt1.shape == dt2.shape;
                break;

                default:
                assert(false);
                return false;
            }
        }

    private:
        Range m_argsrange;

        enum class ComparisonType {
            All = 0,
            Shape = 1,
        };

        ComparisonType m_comparison;
    };

    template<typename FloatType>
    class PassTypeT : TypeClassT<FloatType> {
    private:
        using BaseClass = TypeClassT<FloatType>;
        using SelfClass = PassTypeT<FloatType>;

    public:
        using TypesFunctionArgs = typename BaseClass::TypesFunctionArgs;

        PassTypeT(Range argsrange, Range retsrange) : m_argsrange(argsrange), m_retsrange(retsrange) {
            if( !m_argsrange.singular() ){
                throw std::runtime_error(fmt::format("Expect singular argsrange, got {}->{}", m_argsrange.getBegin(), m_argsrange.getEnd()));
            }
        }
        PassTypeT(const SelfClass& other) = default;

        void check(TypesFunctionArgs& fargs){
            auto& args = fargs.args;
            auto& rets = fargs.rets;
            auto& typeToPass=args[m_argsrange.getBeginAbs(args.size())];
            for(auto ridx: m_retsrange.iterateSafe(rets.size())){
                rets[ridx]=typeToPass;
            }
        }

    private:
        Range m_argsrange;
        Range m_retsrange;
    };
}
