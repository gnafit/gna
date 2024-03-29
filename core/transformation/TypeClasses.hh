#pragma once

#include <boost/range/irange.hpp>
#include <stdexcept>
#include <vector>
#include <string>
#include <fmt/format.h>
#include <fmt/printf.h>

#include "TransformationFunctionArgs.hh"

namespace TypeClasses{
    class Range {
    public:
        /// End is inclusive
        Range(int begin)                          : m_begin(begin), m_end(begin), m_step(1u) {}
        Range(int begin, int end, size_t step=1u) : m_begin(begin), m_end(end), m_step(step) {}
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

        decltype(auto) iterate(size_t size, bool strict=true) const {
            size_t begin, end;
            getRangeAbs(size, begin, end, strict);
            if(begin==-1lu){
                return boost::irange(begin, end, m_step);
            }
            return boost::irange(begin, end+1, m_step);
        }

        decltype(auto) iterateSafe(size_t size) const { return iterate(size, false); }

        std::vector<size_t> vector(size_t size, bool strict=true) const {
            auto it=iterate(size, strict);
            return std::vector<size_t>(it.begin(), it.end());
        }

        void dump(){
            printf("Range %i->%i, step %zu", m_begin, m_end, m_step);
            if(singular()){
                fmt::print(" (singular)");
            }
        }

        void dump(size_t size){
            printf("Range %i->%i, step %zu", m_begin, m_end, m_step);
            if(singular()){
                fmt::print(" (singular)");
            }
            size_t begin, end;
            getRangeAbs(size, begin, end, false);
            fmt::print(" for size {}: {}->{}", size, begin, end);
        }
    private:
        void getRangeAbs(int size, size_t &begin, size_t &end, bool strict=true) const {
            int absbegin = m_begin>=0 ? m_begin : size+m_begin;
            int absend   = m_end>=0   ? m_end   : size+m_end;

            bool error=absend<absbegin && !((m_begin>=0) ^ (m_end>=0));
            if (absend<0 || absbegin>=size || absend<absbegin){
                begin=end=-1lu;
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
        size_t m_step;
    };

    template<typename FloatType>
    class TypeClassT {
    public:
        using TypesFunctionArgs = TransformationTypes::TypesFunctionArgsT<FloatType,FloatType>;
        virtual void processTypes(TypesFunctionArgs& fargs) = 0;
        virtual ~TypeClassT() = default;
    };

    template<typename FloatType>
    class CheckSameTypesT : public TypeClassT<FloatType> {
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
            else if (what=="kind"){
                m_comparison = ComparisonType::Kind;
            }
            else{
                throw std::runtime_error(fmt::format("Unknown comparison type: {}", what));
            }
        }
        CheckSameTypesT(const SelfClass& other) = default;

        void processTypes(TypesFunctionArgs& fargs){
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
                    compare_to->dump();
                    args[aidx].dump();
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

                case ComparisonType::Kind:
                return dt1.kind == dt2.kind;
                break;

                default:
                assert(false);
                return false;
            }
        }

        void dump(){
            const char* names[] = {"types", "shapes", "kinds"};
            fmt::print("TypeClass check same {}: args ", names[static_cast<size_t>(m_comparison)]);
            m_argsrange.dump();
        }
    private:
        Range m_argsrange;

        enum class ComparisonType {
            All = 0,
            Shape = 1,
            Kind = 2,
        };

        ComparisonType m_comparison;
    };

    template<typename FloatType>
    class CheckNdimT : public TypeClassT<FloatType> {
    private:
        using BaseClass = TypeClassT<FloatType>;
        using SelfClass = CheckNdimT<FloatType>;

    public:
        using TypesFunctionArgs = typename BaseClass::TypesFunctionArgs;

        CheckNdimT(size_t ndim, Range argsrange={0,-1}) : m_argsrange(argsrange), m_ndim(ndim) { }
        CheckNdimT(const SelfClass& other) = default;

        void processTypes(TypesFunctionArgs& fargs){
            auto& args = fargs.args;
            for(auto aidx: m_argsrange.iterate(args.size())){
                if(args[aidx].shape.size()!=m_ndim){
                    auto msg = fmt::format("Transformation {0}: input {1} should have dimension {2}, got {3}", args.name(), aidx, m_ndim, args[aidx].shape.size());
                    throw args.error(args[aidx], msg);
                }
            }
        }

        void dump(){
            fmt::print("TypeClass to check ndim is {} ", m_ndim);
            m_argsrange.dump();
        }
    private:
        Range m_argsrange;
        size_t m_ndim;
    };

    template<typename FloatType>
    class CheckKindT : public TypeClassT<FloatType> {
    private:
        using BaseClass = TypeClassT<FloatType>;
        using SelfClass = CheckKindT<FloatType>;

    public:
        using TypesFunctionArgs = typename BaseClass::TypesFunctionArgs;

        CheckKindT(DataKind kind, Range argsrange={0,-1}) : m_kind(kind), m_argsrange(argsrange) { }
        CheckKindT(const SelfClass& other) = default;

        void processTypes(TypesFunctionArgs& fargs){
            static const char* names[] = {"Undefined", "Points", "Hist"};

            auto& args = fargs.args;
            for(auto aidx: m_argsrange.iterate(args.size())){
                if(args[aidx].kind!=m_kind){
                    auto msg = fmt::format("Transformation {0}: input {1} should have kind {2}, got {3}", args.name(), aidx, names[int(m_kind)], names[int(args[aidx].kind)]);
                    throw args.error(args[aidx], msg);
                }
            }
        }

        void dump(){
            fmt::print("TypeClass to check kind is {} ", size_t(m_kind));
            m_argsrange.dump();
        }
    private:
        DataKind m_kind;
        Range m_argsrange;
    };

    template<typename FloatType>
    class PassTypeT : public TypeClassT<FloatType> {
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

        void processTypes(TypesFunctionArgs& fargs){
            auto& args = fargs.args;
            auto& rets = fargs.rets;
            auto& typeToPass=args[m_argsrange.getBeginAbs(args.size())];
            for(auto ridx: m_retsrange.iterateSafe(rets.size())){
                rets[ridx]=typeToPass;
            }
        }

        void dump(){
            fmt::print("TypeClass pass type: args ");
            m_argsrange.dump();
            fmt::print("rets ");
            m_retsrange.dump();
        }
    private:
        Range m_argsrange;
        Range m_retsrange;
    };

    template<typename FloatType>
    class PassTypePriorityT : public TypeClassT<FloatType> {
    private:
        using BaseClass = TypeClassT<FloatType>;
        using SelfClass = PassTypePriorityT<FloatType>;

    public:
        using TypesFunctionArgs = typename BaseClass::TypesFunctionArgs;

        PassTypePriorityT(Range argsrange, Range retsrange, bool prefer_hist=true, bool prefer_nonsingular=true) :
            m_argsrange(argsrange),
            m_retsrange(retsrange),
            m_prefer_hist(prefer_hist),
            m_prefer_nonsingular(prefer_nonsingular)
            { }
        PassTypePriorityT(const SelfClass& other) = default;

        void processTypes(TypesFunctionArgs& fargs){
            /* Cases:
               - 1P - singular Points
               - 1H - singular Hist
               - NP/ND - nonsingular Points/Hist
               - pp - prefer points (true)
               - ph - prefer hist (true)
             Set current:
             | Current \ Next | 1P | 1H | NP | NH       |
             |----------------|----|----|----|----------|
             | 1P             | x  | ph | pn | pn or ph |
             | 1H             | x  | x  | pn | pn       |
             | NP             | x  | x  | x  | ph       |
             | NH             | x  | x  | x  | x        |
             |----------------|----|----|----|----------|
             Do not set current (inverse):
             | Current \ Next | 1P | 1H  | NP  | NH          |
             |----------------|----|-----|-----|-------------|
             | 1P             | v  | !ph | !pn | !pn and !ph |
             | 1H             | v  | v   | !pn | !pn         |
             | NP             | v  | v   | v   | !ph         |
             | NH             | v  | v   | v   | v           |
             |----------------|----|-----|-----|-------------|
             Loop to next, state before current is updated:
             |---------|----------|
             | Current | Loop if  |
             |---------|----------|
             | 1P      | pn or ph |
             | 1H      | pn or ph |
             | NP      | ph       |
             | NH      | x        |
             |---------|----------|
             */
            auto& args = fargs.args;
            DataType const * currentType=&args[m_argsrange.getBeginAbs(args.size())];
            DataType const * nextType;
            for(auto aidx: m_argsrange.iterate(args.size())){
                auto current_ishist = currentType->kind==DataKind::Hist;
                auto current_isnonsingular = currentType->size()>1;
                if(current_ishist){// is Hist
                    if(current_isnonsingular || !m_prefer_nonsingular) {
                        break;
                    }
                }
                else{// is Points
                    if(current_isnonsingular){ // non singular
                        if (!m_prefer_hist){
                            break;
                        }
                    }
                    else{ // singular
                        if (!m_prefer_hist && !m_prefer_nonsingular){
                            break;
                        }
                    }
                }

                nextType=&args[aidx];
                auto next_ishist = nextType->kind==DataKind::Hist;
                auto next_isnonsingular = nextType->size()>1;

                if(current_ishist==next_ishist && current_isnonsingular==next_isnonsingular){
                    continue;
                }
                if(current_isnonsingular && !next_isnonsingular){
                    continue;
                }
                if(current_ishist && !next_isnonsingular){
                    continue;
                }
                if(!m_prefer_nonsingular){
                    if(current_ishist){
                        continue;
                    }
                    if(!current_isnonsingular){
                        if(!next_ishist){
                            continue;
                        }
                        if(!current_isnonsingular && next_isnonsingular){
                            if(next_ishist && !m_prefer_hist){
                                continue;
                            }
                            if(next_ishist){
                                continue;
                            }
                        }

                    }
                }
                if(!m_prefer_hist && current_isnonsingular==next_isnonsingular && !current_ishist && next_ishist){
                    continue;
                }
                currentType = nextType;
            }
            auto& rets = fargs.rets;
            for(auto ridx: m_retsrange.iterateSafe(rets.size())){
                rets[ridx]=*currentType;
            }
        }

        void dump(){
            fmt::print("TypeClass pass type with priority");
            if(m_prefer_nonsingular){
                fmt::print(" [nonsingular]");
            }
            if(m_prefer_hist){
                fmt::print(" [hist]");
            }
            fmt::print(": args ");
            m_argsrange.dump();
            fmt::print(" rets ");
            m_retsrange.dump();
        }
    private:
        Range m_argsrange;
        Range m_retsrange;
        bool m_prefer_hist;
        bool m_prefer_nonsingular;
    };

    template<typename FloatType>
    class SetPointsT : public TypeClassT<FloatType> {
    private:
        using BaseClass = TypeClassT<FloatType>;
        using SelfClass = SetPointsT<FloatType>;

    public:
        using TypesFunctionArgs = typename BaseClass::TypesFunctionArgs;

        SetPointsT(size_t size, Range retsrange={0,-1}) : m_shape({size}), m_retsrange(retsrange) { }
        SetPointsT(size_t sizex, size_t sizey, Range retsrange={0,-1}) : m_shape({sizex, sizey}), m_retsrange(retsrange) { }
        SetPointsT(const SelfClass& other) = default;

        void processTypes(TypesFunctionArgs& fargs){
            auto& rets = fargs.rets;
            for(auto ridx: m_retsrange.iterateSafe(rets.size())){
                rets[ridx].points().shape(m_shape);
            }
        }

        void dump(){
            fmt::print("TypeClass set points: rets ");
            m_retsrange.dump();
        }
    private:
        std::vector<size_t> m_shape;
        Range m_retsrange;
    };

    template<typename FloatType>
    class PassEachTypeT : public TypeClassT<FloatType> {
    private:
        using BaseClass = TypeClassT<FloatType>;
        using SelfClass = PassEachTypeT<FloatType>;

    public:
        using TypesFunctionArgs = typename BaseClass::TypesFunctionArgs;

        PassEachTypeT(Range argsrange={0,-1}, Range retsrange={0,-1}) : m_argsrange(argsrange), m_retsrange(retsrange) { }
        PassEachTypeT(const SelfClass& other) = default;

        void processTypes(TypesFunctionArgs& fargs){
            auto& args = fargs.args;
            auto& rets = fargs.rets;

            auto arange = m_argsrange.iterateSafe(args.size());
            auto rrange = m_retsrange.iterateSafe(rets.size());

            auto rit = rrange.begin();
            for(auto aidx: arange){
                if(rit==rrange.end()){
                    break;
                    //error(args.size(), rets.size());
                }
                rets[*rit]=args[aidx];
                std::advance(rit, 1);
            }
            //if(rit!=rrange.end()){
                //error(args.size(), rets.size());
            //}
        }

        void dump(){
            fmt::print("TypeClass pass each type: args ");
            m_argsrange.dump();
            fmt::print("rets ");
            m_retsrange.dump();
        }
    private:
        void error(size_t nargs, size_t nrets){
            fmt::print("Args: ");
            m_argsrange.dump(nargs);
            fmt::print("\nRets: ");
            m_retsrange.dump(nrets);
            fmt::print("\n");
            throw std::runtime_error("Inconsistent ranges\n");
        }
        Range m_argsrange;
        Range m_retsrange;
    };
}
