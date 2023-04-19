#include "DebugTransformation.hh"

#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include "fmt/format.h"

template<typename T>
std::string message_format(const std::string& message, T arg){
    auto text=message;
    if(text.find("{")!=std::string::npos){
        text = fmt::format(text, arg);
    }
    return text;
}

void message_print(const std::string& name, const std::string& message, bool endl=true){
    std::cout<<message;
    if(name.size()) {
        std::cout<<" ["<<name<<"]";
    }
    if(endl){
        std::cout<<std::endl;
    }
    else{
        std::cout<<std::flush;
    }
}

template<typename T>
void fmtprint(const std::string& name, const std::string& message, T arg, bool endl=true){
    if(message.size()==0u){
        return;
    }
    message_print(name, message_format<>(message, arg), endl);
}


DebugTransformation::DebugTransformation(const std::string& a_name, double a_sleep_seconds):
name(a_name),
sleep_seconds(a_sleep_seconds)
{
    add_transformation();
}

TransformationDescriptor DebugTransformation::add_transformation(){
    std::string tname="debug";
    if(transformations.size()>0){
        tname = fmt::format("{0}_{1:02d}", tname, transformations.size()+1);
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
        auto ninputs=trans.inputs.size()+1;
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
    fmtprint<>(name, message_types, count_types);
}

void DebugTransformation::function(FunctionArgs& fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;
    for (size_t i = 0; i < args.size(); ++i) {
        rets[i].x=args[i].x;
    }

    ++count_function;
    fmtprint<>(name, message, count_function);
    if(sleep_seconds) {
        fmtprint<>("", sleep_message_pre, sleep_seconds, false);
        std::this_thread::sleep_for(std::chrono::nanoseconds(static_cast<size_t>(sleep_seconds*1.e9)));
        fmtprint<>(name, sleep_message_post, sleep_seconds, true);
    }

    if(emit_calculation_error){
        throw rets.error("Test calculation error");
    }
}


