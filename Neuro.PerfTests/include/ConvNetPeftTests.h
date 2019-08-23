#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class ConvNetPeftTests
{
public:
	static void Run()
	{
		Tensor::SetDefaultOpMode(EOpMode::MultiCPU);

        Shape inputShape(64, 64, 4);
        auto model = new Sequential();
        model->AddLayer(new Conv2D(inputShape, 8, 32, 2, 0, new ELU(1)));
        model->AddLayer(new Conv2D(model->LastLayer(), 4, 64, 2, 0, new ELU(1)));
        model->AddLayer(new Conv2D(model->LastLayer(), 4, 128, 2, 0, new ELU(1)));
        model->AddLayer(new Flatten(model->LastLayer()));
        model->AddLayer(new Dense(model->LastLayer(), 512, new ELU(1)));
        model->AddLayer(new Dense(model->LastLayer(), 3, new Softmax()));
        auto net = new NeuralNetwork(model, "test");
        net->Optimize(new Adam(), new Huber(1));

        auto input = Tensor(Shape(64, 64, 4, 32)); input.FillWithRand();
        auto output = Tensor(Shape(1, 3, 1, 32));
        for (int n = 0; n < output.Batch(); ++n)
            output(0, Rng.Next(output.Height()), 0, n) = 1.0f;

        Stopwatch timer;
        timer.Start();

        net->Fit(input, output, 10, 1, nullptr, nullptr, 1, Track::Nothing);

        timer.Stop();
        cout << "Training time " << timer.ElapsedMiliseconds() << "ms";

		return;
    }
};
