#include "DebugTransformation.hh"
#include "fmt/format.h"

#include <thread>
#include <chrono>
#include <iostream>

DebugTransformation::DebugTransformation(){
    add_transformation();
}

DebugTransformation::DebugTransformation(const std::string& a_message, const std::string& a_message_types, double a_sleep_seconds) :
message(a_message),
message_types(a_message_types),
sleep_seconds(a_sleep_seconds)
{
    if(sleep_seconds){
        sleep_message_pre=fmt::format("Sleep for {0:f} seconds...", sleep_seconds);
    }
    add_transformation();
}


TransformationDescriptor DebugTransformation::add_transformation(){
    std::string tname="debug";
    if(transformations.size()>0){
        tname = fmt::format("{0}_{1:02d}", tname, transformations.size());
    }
    transformation_(tname)
        .input("source")
        .output("target")
        .types(TypesFunctions::passAll)
        .types(&DebugTransformation::typesFunction)
        .func(&DebugTransformation::function)
        ;

    return transformations.back();
}

InputDescriptor DebugTransformation::add_input(){
    auto trans=transformations.back();
    auto input=trans.inputs.back();
    if(input.bound()){
        auto ninputs=trans.inputs.size();
        input=trans.input(fmt::format("{0}_{1:02d}", "source", ninputs));
        trans.output(fmt::format("{0}_{1:02d}", "target", ninputs));
    }

    return input;
}

OutputDescriptor DebugTransformation::add_input(OutputDescriptor output){
    auto input=add_input();
    input(output);
    return transformations.back().outputs.back();
}

void DebugTransformation::typesFunction(TypesFunctionArgs& fargs){
    ++count_types;
    if(message_types.size()){
        std::cout<<message_types<<std::endl;
    }
}

void DebugTransformation::function(FunctionArgs& fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;
    for (size_t i = 0; i < args.size(); ++i) {
        rets[i].x=args[i].x;
    }

    ++count_function;
    if(message.size()){
        std::cout<<message<<std::endl;
    }

    if(sleep_seconds) {
        if(sleep_message_pre.size()){
            std::cout<<sleep_message_pre<<std::flush;
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds(static_cast<size_t>(sleep_seconds*1.e9)));

        if(sleep_message_post.size()){
            std::cout<<sleep_message_post<<std::endl;
        }
    }

}
