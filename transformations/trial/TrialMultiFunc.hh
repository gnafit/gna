#pragma once

#include <stdio.h>
#include "TypesFunctions.hh"
#include <vector>

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
      .input("inp")
      .output("out")
      .types(TypesFunctions::passAll)
      .func([](FunctionArgs fargs){
            printf("Calculate: \n");
            printf("  write input to output\n");
            fargs.rets[0].x = fargs.args[0].x;
            printf("  input: ");
            cout<<fargs.args[0].x<<endl;
            printf("  output: ");
            cout<<fargs.rets[0].x<<endl;
            })
      .storage([](TypesFunctionArgs fargs){
             printf("Initialize main storage (nothing): %i\n", (int)fargs.ints.size());
             })
      .func("secondary", [](FunctionArgs fargs){
            printf("Calculate: \n");
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
      .storage("secondary", [](TypesFunctionArgs fargs){
            printf("Initialize secondary storage (clone arg[0]): %i\n", (int)fargs.ints.size());
            fargs.ints[0] = fargs.args[0];
            printf("  after: %i\n", (int)fargs.ints.size());
            })
      .func("thirdparty", [](FunctionArgs fargs){
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
      .storage("thirdparty", [](TypesFunctionArgs fargs){
            printf("Initialize secondary storage (clone arg[0], make 1x1 points): %i\n", (int)fargs.ints.size());
            fargs.ints[0] = fargs.args[0];
            fargs.ints[1] = DataType().points().shape(1);
            printf("  after: %i\n", (int)fargs.ints.size());
            });
  };
};
