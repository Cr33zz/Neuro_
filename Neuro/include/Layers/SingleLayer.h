#pragma once

#include <string>
#include <vector>
#include <map>

#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "ParametersAndGradients.h"
#include "Stopwatch.h"

namespace Neuro
{
    using namespace std;

    class ActivationBase;

    class SingleLayer : public LayerBase
    {
    public:
        virtual ~SingleLayer() {}

        virtual const vector<Shape>& InputShapes() const override { return m_InputShapes; }
        virtual const tensor_ptr_vec_t& Inputs() const override { return m_Inputs; }
        virtual vector<Tensor>& InputsGradient() override { return m_InputsGradient; }
        virtual const vector<Tensor>& Outputs() const override { return m_Outputs; }
        virtual const vector<Shape>& OutputShapes() const override { return m_OutputShapes; }
        virtual const ActivationBase* Activation() const override { return m_Activation; }
        virtual const vector<LayerBase*>& InputLayers() const override { return m_InputLayers; }
        virtual const vector<LayerBase*>& OutputLayers() const override { return m_OutputLayers; }

    protected:
        SingleLayer(const string& constructorName, LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const vector<Shape>& inputShapes, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer() {}

        virtual vector<Shape>& InputShapes() override { return m_InputShapes; }
        virtual tensor_ptr_vec_t& Inputs() override { return m_Inputs; }
        virtual vector<Tensor>& Outputs() override { return m_Outputs; }
        virtual vector<Shape>& OutputShapes() override { return m_OutputShapes; }
        virtual vector<LayerBase*>& InputLayers() override { return m_InputLayers; }
        virtual vector<LayerBase*>& OutputLayers() override { return m_OutputLayers; }

        virtual void OnClone(const LayerBase& source) override;

        vector<const Tensor*> m_Inputs;
        vector<Shape> m_InputShapes;
        vector<LayerBase*> m_InputLayers;
        vector<Tensor> m_InputsGradient;
        // Only models can have multiple outputs
        vector<Tensor> m_Outputs;
        // Only models can have multiple outputs shapes
        vector<Shape> m_OutputShapes;
        vector<LayerBase*> m_OutputLayers;

    private:
        ActivationBase* m_Activation;
    };
}
