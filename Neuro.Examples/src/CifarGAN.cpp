#include "CifarGAN.h"

//////////////////////////////////////////////////////////////////////////
void CifarGAN::LoadImages(Tensor& images)
{
    Tensor labels;
    LoadCifar10Data("data/cifar10_data.bin", images, labels, false);
}

//////////////////////////////////////////////////////////////////////////
Neuro::ModelBase* CifarGAN::CreateGenerator(uint32_t inputsNum)
{
    auto model = new Sequential("generator");
    model->AddLayer(new Dense(100, 256 * 4 * 4, new LeakyReLU(0.2f)));
    model->AddLayer(new Reshape(Shape(4, 4, 256)));
    model->AddLayer(new Conv2DTranspose(128, 4, 2, 1, new LeakyReLU(0.2f)));
    model->AddLayer(new Conv2DTranspose(128, 4, 2, 1, new LeakyReLU(0.2f)));
    model->AddLayer(new Conv2DTranspose(128, 4, 2, 1, new LeakyReLU(0.2f)));
    model->AddLayer(new Conv2D(3, 3, 1, Tensor::GetPadding(Same, 3), new Tanh()));
    return model;
}

//////////////////////////////////////////////////////////////////////////
Neuro::ModelBase* CifarGAN::CreateDiscriminator()
{
    auto model = new Sequential("discriminator");
    model->AddLayer(new Conv2D(Shape(32, 32, 3), 64, 3, 2, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
    model->AddLayer(new Conv2D(128, 3, 2, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
    model->AddLayer(new Conv2D(128, 3, 2, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
    model->AddLayer(new Conv2D(256, 3, 1, Tensor::GetPadding(Same, 3), new LeakyReLU(0.2f)));
    model->AddLayer(new Flatten());
    model->AddLayer(new Dropout(0.4f));
    model->AddLayer(new Dense(1, new Sigmoid()));
    model->Optimize(new Adam(0.0002f, 0.5f), new BinaryCrossEntropy());
    return model;
}
