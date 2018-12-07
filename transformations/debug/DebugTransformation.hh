#pragma once

#include <iostream>
#include "GNAObject.hh"
#include "TypesFunctions.hh"

//
// DebugTransformation transformation
//
class DebugTransformation: public GNASingleObject,
                           public TransformationBind<DebugTransformation> {
public:
    DebugTransformation();
    DebugTransformation(const std::string& a_message, const std::string& a_message_types="", double a_sleep_seconds=0.0);

    TransformationDescriptor add_transformation();
    InputDescriptor          add_input();
    OutputDescriptor         add_input(OutputDescriptor output);

    void typesFunction(TypesFunctionArgs& fargs);
    void function(FunctionArgs& fargs);

    std::string message="Executing transformation function";
    std::string message_types="Executing types function";
    std::string sleep_message_pre="Sleep...";
    std::string sleep_message_post=" done.";
    double      sleep_seconds=0.0;

    size_t count_function=0;
    size_t count_types=0;
};
