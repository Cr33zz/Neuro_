#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Neuro.h"
#include "SimpleGAN.h"

using namespace std;
using namespace Neuro;

class DeepConvGAN : public SimpleGAN
{
protected:
    virtual ModelBase* CreateGenerator(uint32_t inputsNum) override;
    virtual ModelBase* CreateDiscriminator() override;
    virtual string Name() const override { return "mnist_dc_gan"; }
};
