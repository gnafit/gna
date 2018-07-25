#pragma once

#include <iostream>
#include <vector>
#include "GNAObject.hh"
#include "TypesFunctions.hh"

using std::cout;
using std::endl;

/**
 * @brief TrialMultiFunc transformation for testing switchable functions
 *
 * @author Maxim Gonchar
 * @date 18.07.2018
 */
class TrialMultiFunc: public GNASingleObject,
                      public TransformationBind<TrialMultiFunc> {
public:
  TrialMultiFunc(){
    transformation_("multifunc")
      // Any input
      .input("inp")
      // Output of the same shape as input
      .output("out")
      // TypesFunction to define output shape
      .types(TypesFunctions::passAll)
      // TypesFunction to define storage, common for all implementations
      .types([](TypesFunctionArgs& fargs){

             })
      // MemTypesFunction to define another storage, common for all implementations
      .types(&TrialMultiFunc::memTypesFunction)
      // Main Function implementation
      .func([](FunctionArgs& fargs){
            printf("Calculate Function: \n");
            printf("  write input to output\n");
            fargs.rets[0].x = fargs.args[0].x;
            printf("  input: ");
            cout<<fargs.args[0].x<<endl;
            printf("  output: ");
            cout<<fargs.rets[0].x<<endl;
            })
      // Main Function's StorageTypesFunction
      .storage([](StorageTypesFunctionArgs& fargs){
             printf("Initialize main storage (nothing): %i\n", (int)fargs.ints.size());
             })
      // Secondary Function implementation
      .func("secondary", [](FunctionArgs& fargs){
            printf("Calculate secondary Function: \n");
            printf("  write input to output, x2\n");
            printf("  write input to storage\n");
            fargs.rets[0].x = fargs.args[0].x*2;
            fargs.ints[0].x = fargs.args[0].x;
            printf("  input: ");
            cout<<fargs.args[0].x<<endl;
            printf("  output: ");
            cout<<fargs.rets[0].x<<endl;
            printf("  storage: ");
            cout<<fargs.ints[0].x<<endl;
            })
      // Secondary Function StorageTypesFunction
      .storage("secondary", [](StorageTypesFunctionArgs& fargs){
            printf("Initialize secondary storage (clone arg[0]): %i\n", (int)fargs.ints.size());
            fargs.ints[0] = fargs.args[0];
            printf("  after: %i\n", (int)fargs.ints.size());
            })
      // Secondary MemFunction implementation
      .func("secondaryMem", &TrialMultiFunc::memFunction)
      // Secondary MemStorageTypesFunction
      .storage("secondaryMem", &TrialMultiFunc::memStorageTypesFunction)
      // Third party Function implementation
      .func("thirdparty", [](FunctionArgs& fargs){
            printf("Calculate: \n");
            printf("  write input to storage 0\n");
            printf("  write 3 to storage 1\n");
            printf("  write input to output, multiply by storage 1\n");
            fargs.ints[0].x    = fargs.args[0].x;
            fargs.ints[1].x(0) = 3;
            fargs.rets[0].x = fargs.ints[0].x*fargs.ints[1].x(0);
            printf("  input: ");
            cout<<fargs.args[0].x<<endl;
            printf("  storage: ");
            cout<<fargs.ints[0].x<<" and ";
            cout<<fargs.ints[1].x<<endl;
            printf("  output: ");
            cout<<fargs.rets[0].x<<endl;
            })
      // Third party Function StorageTypesFunction
      .storage("thirdparty", [](StorageTypesFunctionArgs& fargs){
            printf("Initialize secondary storage (clone arg[0], make 1x1 points): %i\n", (int)fargs.ints.size());
            fargs.ints[0] = fargs.args[0];
            fargs.ints[1] = DataType().points().shape(1);
            printf("  after: %i\n", (int)fargs.ints.size());
            });
  };

  void memFunction(FunctionArgs& fargs){

  }

  void memTypesFunction(TypesFunctionArgs& fargs){

  }

  void memStorageTypesFunction(StorageTypesFunctionArgs& fargs){

  }
};
