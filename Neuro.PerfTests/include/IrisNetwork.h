#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class IrisNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(Tensor::EOpMode::MultiCPU);

        Shape inputShape(64, 64, 4);
        auto model = new Sequential();
        model->AddLayer(new Dense(4, 1000, new ReLU()));
        model->AddLayer(new Dense(model->LastLayer(), 500, new ReLU()));
        model->AddLayer(new Dense(model->LastLayer(), 300, new ReLU()));
        model->AddLayer(new Dropout(model->LastLayer(), 0.2f));
        model->AddLayer(new Dense(model->LastLayer(), 3, new Softmax()));
        auto net = new NeuralNetwork(model, "test");
        net->Optimize(new Adam(), new CrossEntropy());

        vector<tensor_ptr_vec_t> inputs;
        vector<tensor_ptr_vec_t> outputs;

        LoadCSVData("iris_data.csv", 1, inputs, outputs, true);

        Stopwatch timer;
        timer.Start();

        net->Fit(inputs, outputs, 20, 10, nullptr, nullptr, 1, Track::TrainError);

        timer.Stop();
        cout << "Training time " << timer.ElapsedMiliseconds() << "ms";

        return;
    }
};
