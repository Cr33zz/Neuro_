#include "DeepConvGAN.h"

//////////////////////////////////////////////////////////////////////////
ModelBase* DeepConvGAN::CreateGenerator(uint32_t inputsNum)
{
    auto model = new Sequential("generator");
    model->AddLayer(new Dense(inputsNum, 128 * 7 * 7, new ReLU()));
    model->AddLayer(new Reshape(Shape(7, 7, 128)));
    model->AddLayer(new UpSampling2D(2));
    model->AddLayer(new Conv2D(3, 128, 1, Tensor::GetPadding(Same, 3)));
    model->AddLayer((new BatchNormalization())->SetMomentum(0.8f));
    model->AddLayer(new Activation(new ReLU()));
    model->AddLayer(new UpSampling2D(2));
    model->AddLayer(new Conv2D(3, 64, 1, Tensor::GetPadding(Same, 3)));
    model->AddLayer((new BatchNormalization())->SetMomentum(0.8f));
    model->AddLayer(new Activation(new ReLU()));
    model->AddLayer(new Conv2D(3, 1, 1, Tensor::GetPadding(Same, 3), new Tanh()));
    return model;
}

//////////////////////////////////////////////////////////////////////////
ModelBase* DeepConvGAN::CreateDiscriminator()
{
    auto model = new Sequential("discriminator");
    model->AddLayer(new Conv2D(Shape(28, 28, 1), 3, 32, 2, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
    model->AddLayer(new Dropout(0.25f));
    model->AddLayer(new Conv2D(3, 64, 2, Tensor::GetPadding(Same, 3)));
    model->AddLayer((new BatchNormalization())->SetMomentum(0.8f));
    model->AddLayer(new Activation(new LeakyReLU(0.2f)));
    model->AddLayer(new Dropout(0.25f));
    model->AddLayer(new Conv2D(3, 128, 2, Tensor::GetPadding(Same, 3)));
    model->AddLayer((new BatchNormalization())->SetMomentum(0.8f));
    model->AddLayer(new Activation(new LeakyReLU(0.2f)));
    model->AddLayer(new Dropout(0.25f));
    model->AddLayer(new Conv2D(3, 256, 1, Tensor::GetPadding(Same, 3)));
    model->AddLayer((new BatchNormalization())->SetMomentum(0.8f));
    model->AddLayer(new Activation(new LeakyReLU(0.2f)));
    model->AddLayer(new Dropout(0.25f));
    model->AddLayer(new Flatten());
    model->AddLayer(new Dense(1, new Sigmoid()));
    model->Optimize(new Adam(0.0002f, 0.5f), new BinaryCrossEntropy());
    return model;
}