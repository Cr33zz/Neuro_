#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "Neuro.h"

using namespace std;
using namespace Neuro;

class GAN
{
public:
    void Run();
    void RunDiscriminatorTrainTest();

protected:
    virtual void LoadImages(Tensor& images);
    virtual ModelBase* CreateGenerator(uint32_t inputsNum);
    virtual ModelBase* CreateDiscriminator();
    virtual string Name() const { return "mnist_vanilla_gan"; }

private:
    Shape m_ImageShape;
};
