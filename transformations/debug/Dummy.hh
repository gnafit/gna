#pragma once

#include <stdio.h>
#include "GNAObject.hh"
#include "TypesFunctions.hh"
#include <vector>

namespace GNA{
    namespace GNAObjectTemplates{
        /**
         * @brief Dummy transformation for testing pupropses.
         *
         * Does nothing. May have any number of inputs/outputs and use variables.
         *
         * @author Maxim Gonchar
         * @date 2017.05
         */
        template<typename FloatType>
        class DummyT: public GNASingleObjectT<FloatType,FloatType>,
                      public TransformationBind<DummyT<FloatType>,FloatType,FloatType> {
        public:
            using DummyType = DummyT<FloatType>;
            using typename GNAObjectT<FloatType,FloatType>::FunctionArgs;
            using typename GNAObjectT<FloatType,FloatType>::SingleOutput;
            using typename GNAObjectT<FloatType,FloatType>::OutputDescriptor;
            using typename GNAObjectT<FloatType,FloatType>::InputDescriptor;

            /**
             * @brief Constructor.
             * @param shape - size of each output.
             * @param label - transformation label.
             * @param labels - variables to bind.
             */
            DummyT(size_t shape, const char* label, const std::vector<std::string> &labels={});

            /** @brief Add an input by name and leave unconnected. */
            InputDescriptor add_input(const std::string& name){
                auto trans = this->transformations.back();
                auto input = trans.input(name);
                return InputDescriptor(input);
            }

            /** @brief Add an input. */
            void add_input(SingleOutput& output, const std::string& name){
                auto out=output.single();
                auto input=add_input(name.size() ? name : out.name());
                output.single() >> input;
            }

            /** @brief Add an output by name */
            OutputDescriptor add_output(const std::string& name){
                auto trans = this->transformations.back();
                auto output = trans.output(name);
                trans.updateTypes();
                return OutputDescriptor(output);
            }

            void dummy_fcn(FunctionArgs& fargs);
            void dummy_gpuargs_h_local(FunctionArgs& fargs);
            void dummy_gpuargs_h(FunctionArgs& fargs);
            void dummy_gpuargs_d(FunctionArgs& fargs);

            std::vector<variable<FloatType>> m_vars;
        };
    }
}
