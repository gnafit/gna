#pragma once

#include <iostream>
#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"

namespace GNA{
    namespace GNAObjectTemplates{
        template<typename FloatType>
        class SnapshotT: public GNAObjectBind1N<FloatType>,
                         public TransformationBind<SnapshotT<FloatType>,FloatType,FloatType> {
        private:
            using BaseClass = GNAObjectT<FloatType,FloatType>;
            using Snapshot = SnapshotT<FloatType>;
        public:
            using typename BaseClass::FunctionArgs;
            using typename BaseClass::SingleOutput;
            using TransformationDescriptor = typename BaseClass::TransformationDescriptorType;

            SnapshotT();
            SnapshotT(SingleOutput& output);

            void nextSample();

            TransformationDescriptor add_transformation(const std::string& name="");

        protected:
            void makeSnapshot(FunctionArgs& fargs);
        };
    }
}
