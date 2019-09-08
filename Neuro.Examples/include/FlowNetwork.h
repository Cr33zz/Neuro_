#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class FlowNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::MultiCPU);

        auto input1 = new Dense(2, 2, new Sigmoid(), "input_1");
        auto upperStream1 = new Dense(input1, 2, new Linear(), "upperStream_1");
        auto lowerStream1 = new Dense(input1, 2, new Linear(), "lowerStream_1");
        auto merge = new Merge({upperStream1, lowerStream1}, Merge::Mode::Sum);
        auto model = new Flow({ input1 }, { merge }, "flow");

        cout << model->Summary();

        model->Optimize(new SGD(0.05f), new MeanSquareError());

        tensor_ptr_vec_t inputs = { new Tensor({ 0, 1 }, Shape(1, 2)) };
        tensor_ptr_vec_t outputs = { new Tensor({ 0, 1 }, Shape(1, 2)) };

        model->Fit(inputs, outputs, 1, 60, nullptr, nullptr, 2, ETrack::Nothing, false);

        cout << model->TrainSummary();
        
        cin.get();
        return;
    }
};
