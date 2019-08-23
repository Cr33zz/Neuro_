#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class AutoencoderNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::MultiCPU);

        //Based on https://www.youtube.com/watch?v=6Lfra0Tym4M

        Shape inputShape(28, 28, 1);
        auto model = new Sequential();
        model->AddLayer(new Conv2D(inputShape, 5, 10, 1, new ReLU()));
        model->AddLayer(new Pooling(model->LastLayer(), 2));
        model->AddLayer(new Conv2D(model->LastLayer(), 2, 20, 1, new ReLU()));
        model->AddLayer(new Pooling(model->LastLayer(), 2));
        model->AddLayer(new UpSampling(model->LastLayer(), 2));
        model->AddLayer(new Deconvolution(model->LastLayer(), 2, 20, 1, new ReLU()));
        model->AddLayer(new UpSampling(model->LastLayer(), 2));
        model->AddLayer(new Deconvolution(model->LastLayer(), 5, 10, 1, new ReLU()));
        model->AddLayer(new Deconvolution(model->LastLayer(), 1, 3, 1, new Sigmoid()));
        auto net = new NeuralNetwork(model, "autoencoder");
        net->Optimize(new Adam(), new BinaryCrossEntropy());

        vector<tensor_ptr_vec_t> inputs;
        vector<tensor_ptr_vec_t> outputs;


        Stopwatch timer;
        timer.Start();

        net->Fit(inputs, outputs, 20, 10, nullptr, nullptr, 1, Track::TrainError);

        timer.Stop();
        cout << "Training time " << timer.ElapsedMiliseconds() << "ms";

        return;
    }
};
