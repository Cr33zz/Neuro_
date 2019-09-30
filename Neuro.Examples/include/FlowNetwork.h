#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class FlowNetwork
{
public:
    void Run()
    {
        Tensor::SetDefaultOpMode(MultiCPU);

        auto input1 = new Dense(2, 2, new Sigmoid(), "input_1");
        auto upperStream1 = new Dense(input1, 2, new Linear(), "upperStream_1");
        auto lowerStream1 = new Dense(input1, 2, new Linear(), "lowerStream_1");
        auto merge = new Merge({upperStream1, lowerStream1}, MergeSum);
        auto model = Flow({ input1 }, { merge }, "flow");

        cout << "Example: " << model.Name() << endl;
        cout << model.Summary();

        model.Optimize(new SGD(0.05f), new MeanSquareError());

        const_tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(2)) };
        const_tensor_ptr_vec_t outputs = { new Tensor({ 0, 1 }, Shape(2)) };

        model.Fit(inputs, outputs, 1, 60, nullptr, nullptr, 2, TrainError, false);

        cout << model.TrainSummary();
        
        cin.get();
        return;
    }
};
