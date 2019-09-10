#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class ConvAutoencoderNetwork
{
public:
    static void Run()
    {
        Tensor::SetDefaultOpMode(EOpMode::GPU);

        //Based on https://blog.keras.io/building-autoencoders-in-keras.html

        auto encoder = Sequential("encoder");
        encoder.AddLayer(new Conv2D(Shape(28, 28, 1), 3, 16, 1, 1, new ReLU()));
        encoder.AddLayer(new MaxPooling2D(2, 2));
        encoder.AddLayer(new Conv2D(3, 8, 1, 1, new ReLU()));
        encoder.AddLayer(new MaxPooling2D(2, 2));
        cout << encoder.Summary();
        auto decoder = Sequential("decoder");
        decoder.AddLayer(new Conv2D(Shape(7, 7, 8), 3, 8, 1, 1, new ReLU()));
        decoder.AddLayer(new UpSampling2D(2));
        decoder.AddLayer(new Conv2D(3, 16, 1, 1, new ReLU()));
        decoder.AddLayer(new UpSampling2D(2));
        decoder.AddLayer(new Conv2D(3, 1, 1, 1, new Sigmoid()));
        cout << decoder.Summary();

        auto model = Sequential("conv_autoencoder");
        model.AddLayer(&encoder);
        model.AddLayer(&decoder);
        model.Optimize(new Adam(), new BinaryCrossEntropy());
        //cout << model.Summary();

        Tensor input, output;
        LoadMnistData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", input, output, false, 6000);

        model.Fit(input, input, 256, 50, nullptr, nullptr, 2, ETrack::TrainError);

        cout << model.TrainSummary();

        vector<uint32_t> samplesIds(36);
        iota(samplesIds.begin(), samplesIds.end(), 0);
        Tensor samples(Shape(28, 28, 1, (uint32_t)samplesIds.size()));
        input.GetBatches(samplesIds, samples);
        Tensor decodedSamples = model.Predict(samples)[0];
        samples.SaveAsImage("original_conv.png", true);
        decodedSamples.SaveAsImage("decoded_conv.png", true);

        cin.get();
        return;
    }
};
