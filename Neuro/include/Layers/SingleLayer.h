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

    class NEURO_DLL_EXPORT SingleLayer : public LayerBase
    {
    public:
        virtual ~SingleLayer();

    protected:
        SingleLayer(const string& constructorName, const Shape& inputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer() {}

        virtual void OnClone(const LayerBase& source) override;
        //virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes) override;

        ActivationBase* m_Activation;
    };
}
