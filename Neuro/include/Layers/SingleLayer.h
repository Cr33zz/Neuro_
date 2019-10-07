#pragma once

#include <string>
#include <vector>
#include <map>

#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "ParameterAndGradient.h"
#include "Stopwatch.h"

namespace Neuro
{
    using namespace std;

    class Variable;
    class TensorLike;
    class ActivationBase;

    class SingleLayer : public LayerBase
    {
    public:
        virtual ~SingleLayer();

    protected:
        SingleLayer(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer() {}

        virtual void OnClone(const LayerBase& source) override;
        virtual void InitOps(TensorLike* training, bool initValues = true);

    private:
        ActivationBase* m_Activation;
    };
}
