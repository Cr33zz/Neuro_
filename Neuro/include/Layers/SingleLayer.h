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

    class ActivationBase;

    class SingleLayer : public LayerBase
    {
    public:
        virtual ~SingleLayer();

        virtual const Shape& InputShape() const override { return m_InputShape; }
        virtual const tensor_ptr_vec_t& Outputs() const override { return m_Outputs; }
        virtual const vector<Shape>& OutputShapes() const override { return m_OutputsShapes; }
        virtual const vector<LayerBase*>& InputLayers() const override { return m_InputLayers; }
        virtual const vector<LayerBase*>& OutputLayers() const override { return m_OutputLayers; }

    protected:
        SingleLayer(const string& constructorName, LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer() {}

        virtual vector<TensorLike*>& InputOps() override { return m_InputOps; }
        virtual vector<TensorLike*>& OutputOps() override { return m_OutputOps; }

        virtual void OnClone(const LayerBase& source) override;
        virtual void OnInit(bool initValues = true) override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
        virtual void OnLinkOutput(LayerBase* outputLayer) override;

        virtual void InitOps(bool initValues = true) {}

        Shape m_InputShape;
        vector<TensorLike*> m_InputOps;
        vector<TensorLike*> m_OutputOps;
        vector<Tensor*> m_Outputs;
        vector<Shape> m_OutputsShapes;
        vector<LayerBase*> m_InputLayers;
        vector<LayerBase*> m_OutputLayers;

    private:
        ActivationBase* m_Activation;
    };
}
