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

protected:
    virtual ModelBase* CreateGenerator(uint32_t inputsNum);
    virtual ModelBase* CreateDiscriminator();
    virtual string Name() const { return "gan"; }
};
