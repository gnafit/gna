#pragma once

#include "GNAObjectBindMN.hh"

namespace GNA{
    namespace GNAObjectTemplates{
        /**
         * @brief Calculate the element-wise sum of the inputs.
         *
         * Unlike Sum transformation, MultiSum support multiple sets of inputs.
         *
         * Outputs:
         *   - `sum.sum` -- the result of a sum.
         *
         * @author Maxim Gonchar
         * @date 05.02.2019
         */
        template <typename FloatType>
        class MultiSumT: public GNAObjectBindMN<FloatType>,
                         public TransformationBind<MultiSumT<FloatType>,FloatType,FloatType> {
        private:
            using BaseClass = GNAObjectT<FloatType,FloatType>;
            using BindClass = GNAObjectBindMN<FloatType>;

            using typename BaseClass::OutputDescriptors;
            using TransformationDescriptor = typename BaseClass::TransformationDescriptorType;
            using BaseClass::transformations;

            using BindClass::new_transformation_name;
            using BindClass::add_transformation;
            using BindClass::add_input;
            using BindClass::add_output;
            using BindClass::set_open_input;
        public:
            using typename BaseClass::FunctionArgs;
            using typename BaseClass::TypesFunctionArgs;

            MultiSumT();                                 ///< Constructor.
            MultiSumT(const OutputDescriptors& outputs); ///< Construct MultiSum from vector of outputs

            TransformationDescriptor add_transformation(const std::string& name=""); ///< Add new transformation
        };
    }
}
