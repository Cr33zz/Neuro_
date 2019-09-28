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

        virtual const tensor_ptr_vec_t& FeedForward(const const_tensor_ptr_vec_t& inputs, bool training) override;
        virtual const tensor_ptr_vec_t& BackProp(const tensor_ptr_vec_t& outputsGradient) override;

        virtual const Shape& InputShape() const override { return m_InputShape; }
        virtual const vector<Tensor*>& InputsGradient() override { return m_InputsGradient; }
        virtual const tensor_ptr_vec_t& Outputs() const override { return m_Outputs; }
        virtual const vector<Shape>& OutputShapes() const override { return m_OutputsShapes; }
        virtual const vector<LayerBase*>& InputLayers() const override { return m_InputLayers; }
        virtual const vector<LayerBase*>& OutputLayers() const override { return m_OutputLayers; }

        virtual int InputOffset(const LayerBase* inputLayer) const override;

    protected:
        SingleLayer(const string& constructorName, LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer() {}

        virtual void OnClone(const LayerBase& source) override;
        virtual void OnInit() override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
        virtual void OnLinkOutput(LayerBase* outputLayer) override;

        virtual void FeedForwardInternal(bool training);
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient);

        Shape m_InputShape;
        const_tensor_ptr_vec_t m_Inputs;
        vector<Tensor*> m_InputsGradient;
        vector<Tensor*> m_Outputs;
        vector<Shape> m_OutputsShapes;
        vector<LayerBase*> m_InputLayers;
        vector<LayerBase*> m_OutputLayers;

    private:
        ActivationBase* m_Activation;
    };
}
