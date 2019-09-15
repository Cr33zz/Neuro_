#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"
#include "GAN.h"

using namespace std;
using namespace Neuro;

class CifarGAN : public GAN
{
protected:
    virtual void LoadImages(Tensor& images);
    virtual ModelBase* CreateGenerator(uint32_t inputsNum) override;
    virtual ModelBase* CreateDiscriminator() override;
    virtual string Name() const override { return "cifar_dc_gan"; }
};

