#pragma once

#include "ComputationalGraph/TensorLike.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    class Optimizer;
    class InitializerBase;

    class Variable : public TensorLike
    {
    public:
        Variable(const Tensor& initValue, const string& name = "");
        Variable(float initValue, const string& name = "");
        Variable(const Shape& shape, InitializerBase* initializer = nullptr, const string& name = "");

        void Trainable(bool enabled) { m_Trainable = enabled; }
        bool Trainable() const { return m_Trainable; }

        void Init();

    private:
        bool m_Trainable = true;
        InitializerBase* m_Initializer = nullptr;
    };
}
