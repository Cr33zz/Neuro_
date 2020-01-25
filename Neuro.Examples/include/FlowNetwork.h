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
        Tensor::SetDefaultOpMode(CPU_MT);
        GlobalRngSeed(1337);

        /*auto input1 = new Input(Shape(2));
        auto upperStream1 = (new Dense(2, new Linear(), "upperStream_1"))->Call(input1->Outputs());
        auto lowerStream1 = (new Dense(2, new Linear(), "lowerStream_1"))->Call(input1->Outputs());
        auto merge = (new Merge(MergeSum))->Call({ upperStream1[0], lowerStream1[0] });
        auto model = new Flow(input1->Outputs(), merge, "flow");

        cout << "Example: " << model->Name() << endl;
        cout << model->Summary();

        model->Optimize(new SGD(0.05f), new MeanSquareError());

        const_tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(2)) };
        const_tensor_ptr_vec_t outputs = { new Tensor({ 0, 1 }, Shape(2)) };

        model->Fit(inputs, outputs, 1, 20, nullptr, nullptr, 2, TrainError, false);

        cout << model->TrainSummary();*/
        
        cin.get();
        return;
    }
};
