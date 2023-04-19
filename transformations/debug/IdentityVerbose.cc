#include "IdentityVerbose.hh"

#include <iostream>
#include "TypeClasses.hh"
using namespace TypeClasses;

using std::cout;
using std::endl;

IdentityVerbose::IdentityVerbose(std::string prefix) :
GNAObjectBind1N<double>("identity", "source", "target", 0, 0, 0),
m_prefix(prefix)
{
    this->add_transformation();
    this->add_input();
    this->set_open_input();
}

void IdentityVerbose::perform(FunctionArgs& fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;
    auto size = args.size();

    // Trigger computation before printing
    for (size_t i = 0; i < args.size(); ++i) {
        args[i];
    }

    if(size>1){
        cout<<m_prefix<<" ("<<size<<"):"<<endl;
    }
    else{
        cout<<m_prefix<<": ";
    }

    for (size_t i = 0; i < args.size(); ++i) {
        auto& arg = args[i];
        rets[i].x = arg.x;

        if(size>1) {
            cout<<i<<": ";
        }
        if(arg.type.shape.size()>1){
            cout<<arg.arr2d<<endl;
        }
        else{
            cout<<arg.x<<endl;
        }
    }
}

TransformationDescriptor IdentityVerbose::add_transformation(const std::string& name){
    this->transformation_(new_transformation_name(name))
        .types(new PassEachTypeT<double>())
        .func(&IdentityVerbose::perform);

    reset_open_input();
    return this->transformations.back();
}
