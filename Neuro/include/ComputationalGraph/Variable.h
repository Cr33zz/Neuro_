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
        explicit Variable(const Tensor& initValue, const string& name = "");
        explicit Variable(float initValue, const string& name = "");
        explicit Variable(const Shape& shape, InitializerBase* initializer = nullptr, const string& name = "");
        virtual ~Variable();

        virtual bool IsVar() const override { return true; }

        void SetTrainable(bool enabled) { m_Trainable = enabled; }
        bool Trainable() const { return m_Trainable; }

        void Initialize();
        void ForceInitialized() { m_Initialized = true; }

        virtual bool CareAboutGradient() const override;

    private:
        bool m_Trainable = true;
        bool m_Initialized = false;
        InitializerBase* m_Initializer = nullptr;
    };
}
