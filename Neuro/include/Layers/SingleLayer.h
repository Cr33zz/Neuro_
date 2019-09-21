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
        virtual ~SingleLayer();

        virtual const tensor_ptr_vec_t& FeedForward(const const_tensor_ptr_vec_t& inputs, bool training) override;
        virtual const tensor_ptr_vec_t& BackProp(const tensor_ptr_vec_t& outputsGradient) override;

        virtual const vector<Shape>& InputShapes() const override { return m_InputsShapes; }
        virtual const vector<Tensor*>& InputsGradient() override { return m_InputsGradient; }
        virtual const tensor_ptr_vec_t& Outputs() const override { return m_Outputs; }
        virtual const vector<Shape>& OutputShapes() const override { return m_OutputsShapes; }
        virtual const vector<LayerBase*>& InputLayers() const override { return m_InputLayers; }
        virtual const vector<LayerBase*>& OutputLayers() const override { return m_OutputLayers; }

        virtual int InputOffset(const LayerBase* inputLayer) const override;

        /*int FeedForwardTime() const { return (int)m_FeedForwardTimer.ElapsedMilliseconds(); }
        int BackPropTime() const { return (int)m_BackPropTimer.ElapsedMilliseconds(); }
        int ActivationTime() const { return (int)m_ActivationTimer.ElapsedMilliseconds(); }
        int ActivationBackPropTime() const { return (int)m_ActivationBackPropTimer.ElapsedMilliseconds(); }*/

    protected:
        SingleLayer(const string& constructorName, LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const vector<Shape>& inputShapes, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer(const string& constructorName, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        SingleLayer() {}

        virtual void OnClone(const LayerBase& source) override;
        virtual void OnInit() override;
        virtual void OnLink(LayerBase* layer, bool input) override;

        virtual void FeedForwardInternal(bool training);
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient);

        const_tensor_ptr_vec_t m_Inputs;
        vector<Shape> m_InputsShapes;
        vector<Tensor*> m_InputsGradient;
        // Only models can have multiple outputs
        vector<Tensor*> m_Outputs;
        // Only models can have multiple outputs shapes
        vector<Shape> m_OutputsShapes;
        vector<LayerBase*> m_InputLayers;
        vector<LayerBase*> m_OutputLayers;

    private:
        ActivationBase* m_Activation;

        Stopwatch m_FeedForwardTimer;
        Stopwatch m_ActivationTimer;
        Stopwatch m_BackPropTimer;
        Stopwatch m_ActivationBackPropTimer;
    };
}
