#pragma once

#include <stdio.h>
#include <vector>
#include <iostream>
#include "fmt/format.h"
#include "GNAObject.hh"
#include "OpenHandle.hh"
#include "TypeClasses.hh"
#include "TransformationErrors.hh"

namespace GNA{
    namespace GNAObjectTemplates{
        /**
         * @brief DummyType transformation for testing pupropses.
         *
         * Does nothing. May have any number of inputs/outputs and use variables. May apply custrom type classes.
         *
         * @author Maxim Gonchar
         * @date 2017.05
         */
        template<typename FloatType>
        class DummyTypeT: public GNAObjectT<FloatType,FloatType>,
                          public TransformationBind<DummyTypeT<FloatType>,FloatType,FloatType> {
        private:
            using BaseClass = GNAObjectT<FloatType,FloatType>;
            using BaseClass::transformations;
            using SelfClass = DummyTypeT<FloatType>;
        public:
            using typename BaseClass::SingleOutput;
            using typename BaseClass::InputDescriptor;
            using typename BaseClass::OutputDescriptor;
            using typename BaseClass::FunctionArgs;
            using typename BaseClass::TypesFunctionArgs;
            using TypeClass = TypeClasses::TypeClassT<FloatType>;
          /**
           * @brief Constructor.
           */
            DummyTypeT(){
              this->transformation_("dummytype").func(&SelfClass::dummytype_fcn).no_autotype();
            }

          /** @brief Add an input by name and leave unconnected. */
          InputDescriptor add_input(const std::string& name=""){
            auto trans = transformations.back();
            return trans.input(name.size() ? name : fmt::format( "input_{:02d}", trans.inputs.size()));
          }

          /** @brief Add an input. */
          void add_input(SingleOutput& output, const std::string& name=""){
            auto input=add_input(name);
            output.single() >> input;
          }

          /** @brief Add an output by name */
          OutputDescriptor add_output(const std::string& name=""){
            auto trans = transformations.back();
            auto output = trans.output(name.size() ? name : fmt::format( "output_{:02d}", trans.outputs.size()));
            trans.updateTypes();
            return OutputDescriptor(output);
          }

          void add_typeclass(TypeClass* tcls){
            auto* entry = OpenHandleT<FloatType,FloatType>(transformations.back()).getEntry();
            entry->typeclasses.push_back(tcls);
          }

          bool process_types(){
              try {
                  OpenHandleT<FloatType,FloatType>(transformations.back()).getEntry()->evaluateTypes();
              }
              catch(const std::runtime_error& ex) {
                  std::cout<<ex.what()<<std::endl;
                  return false;
              }
              return true;
          }

          void dummytype_fcn(FunctionArgs& fargs){
            auto& rets = fargs.rets;
            for (size_t i = 0; i < rets.size(); ++i) {
              rets[i].x=static_cast<double>(i);
            }
          }
        };
    }
}

