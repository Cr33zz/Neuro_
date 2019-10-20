#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class ConvNetwork
{
public:
	void Run()
	{
		Tensor::SetDefaultOpMode(GPU);
        GlobalRngSeed(1337);

        Shape inputShape(64, 64, 4);
        auto model = Sequential("conv");
        model.AddLayer(new Conv2D(inputShape, 32, 8, 2, 0, new ELU(1)));
        model.AddLayer(new Conv2D(64, 4, 2, 0, new ELU(1)));
        model.AddLayer(new Conv2D(128, 4, 2, 0, new ELU(1)));
        model.AddLayer(new Flatten());
        model.AddLayer(new Dense(512, new ELU(1)));
        model.AddLayer(new Dense(3, new Softmax()));

        cout << "Example: " << model.Name() << endl;
        cout << model.Summary();

        model.Optimize(new Adam(), new BinaryCrossEntropy());

        auto input = Tensor(Shape(64, 64, 4, 32)); input.FillWithRand();
        auto output = Tensor(Shape(3, 1, 1, 32));
        for (uint32_t n = 0; n < output.Batch(); ++n)
            output(0, GlobalRng().Next(output.Height()), 0, n) = 1.0f;

        model.Fit(input, output, -1, 10, nullptr, nullptr, 2);

        cout << model.TrainSummary();

        cin.get();
		return;
    }
};
