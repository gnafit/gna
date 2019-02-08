#include "unit_transformations.hh"

#include <vector>
#include <iostream>
#include <utility>

using namespace std;

class Base{
public:

protected:
    Base() = default;
    Base(const Base&) = delete;
};

template<typename FloatType>
class Movable : public Base {
    public:
        Movable() {
            printf("Default consructor %p\n", (void*)this);
            m_vector = {-1,1,10};
        }
        ~Movable(){
            printf("Destructor %p, size %zu\n", (void*)this, m_vector.size());
        }
        //Movable(const Movable& other) {
            //printf("Copy consructor %p\n", (void*)this);
        //}
        Movable(const Movable<FloatType>& other) = delete;
        Movable(Movable<FloatType>&& other) noexcept : m_vector(std::move(other.m_vector)) {
            printf("Move consructor %p\n", (void*)this);
            other.m_vector.resize(0);
        }

        Movable& chain(){
            return *this;
        }

        //Movable&& mchain(){
            //return move(*this);
        //}

        void dump() const {
            printf("Vector at %p ",(void*)this);
            for(auto num: m_vector){
                printf("%i  ", (int)num);
            }
            printf("\n");
        }

        vector<FloatType> m_vector;
};

template<typename FloatType>
Movable<FloatType> fcnVal(){
    return Movable<FloatType>();
}

template<typename FloatType>
Movable<FloatType> fcnMove(){
    return move(Movable<FloatType>());
}

void GNAUnitTest::testMove1(){
    printf("Store fcnVal\n");
    Movable<int>& var1=fcnVal<int>().chain();
    printf("dump");
    var1.dump();
    printf("\n");

    printf("Store mchain fcnVal\n");
    Movable<int>&& var3=fcnVal<int>().mchain();
    printf("dump");
    var3.dump();
    printf("\n");

    printf("Store fcnMove\n");
    const Movable<int>& var2=fcnMove<int>().chain();
    printf("dump");
    var2.dump();
    printf("\n");
}

void GNAUnitTest::testMove(){
    printf("Call fcnVal\n");
    fcnVal<int>();
    printf("\n");

    printf("Call fcnMove\n");
    fcnMove<int>();
    printf("\n");

    printf("Call dump fcnVal\n");
    fcnVal<int>().dump();
    printf("\n");

    printf("Call dump fcnMove\n");
    fcnMove<int>().dump();
    printf("\n");

    printf("Store fcnVal\n");
    Movable<int> var1=fcnVal<int>();
    printf("dump");
    var1.dump();
    printf("\n");

    printf("Store fcnMove\n");
    Movable<int> var2=fcnMove<int>();
    printf("dump");
    var2.dump();
    printf("\n");

    printf("Store ref fcnVal\n");
    Movable<int>&& var3=fcnVal<int>();
    printf("dump");
    var3.dump();
    printf("\n");

    printf("Store ref fcnMove\n");
    Movable<int>&& var4=fcnMove<int>();
    printf("dump");
    var4.dump();
    printf("\n");

    printf("Store ref1 fcnVal\n");
    const Movable<int>& var5=fcnVal<int>();
    printf("dump");
    var5.dump();
    printf("\n");

    printf("Store ref1 fcnMove\n");
    const Movable<int>& var6=fcnMove<int>();
    printf("dump");
    var6.dump();
    printf("\n");

    printf("Store auto fcnVal\n");
    auto var7=fcnVal<int>();
    printf("dump");
    var7.dump();
    printf("\n");

    printf("Store auto fcnMove\n");
    auto var8=fcnMove<int>();
    printf("dump");
    var8.dump();
    printf("\n");

    printf("Store auto& fcnVal\n");
    const auto& var9=fcnVal<int>();
    printf("dump");
    var9.dump();
    printf("\n");

    printf("Store auto& fcnMove\n");
    const auto& var10=fcnMove<int>();
    printf("dump");
    var10.dump();
    printf("\n");

    printf("Store auto&& fcnVal\n");
    auto&& var11=fcnVal<int>();
    printf("dump");
    var11.dump();
    printf("\n");

    printf("Store auto&& fcnMove\n");
    auto&& var12=fcnMove<int>();
    printf("dump");
    var12.dump();
    printf("\n");
}
