#pragma once

#include <map>
#include <vector>

#include "Models/ModelBase.h"

namespace Neuro
{
	using namespace std;

    // Represents a graph of interconnected layers.
    // When linking this model's input to other layer(s) all internal model input layers will have the same inputs shapes and input layers.
    // Number of outputs will be a sum of all internal model output layers. Inputs gradient will be averaged across internal model input layers'
    // inputs gradients.
    class Flow : public ModelBase
    {
	public:
        Flow(const vector<LayerBase*>& inputLayers, const vector<LayerBase*>& outputLayers, const string& name = "", int seed = 0);
        ~Flow();

        virtual const Shape& InputShape() const override { return m_ModelInputLayers[0]->InputShape(); }
        virtual const vector<Tensor*>& InputsGradient() override { return m_InputsGradient; }
        virtual const tensor_ptr_vec_t& Outputs() const override { return m_Outputs; }
        virtual const vector<Shape>& OutputShapes() const override { return m_OutputsShapes; }
        virtual const vector<LayerBase*>& InputLayers() const override { return m_ModelInputLayers[0]->InputLayers(); }
        virtual const vector<LayerBase*>& OutputLayers() const override { return m_ModelOutputLayers[0]->OutputLayers(); }

        virtual int InputOffset(const LayerBase* inputLayer) const override;

        virtual const tensor_ptr_vec_t& FeedForward(const const_tensor_ptr_vec_t& inputs, bool training) override;
        virtual const tensor_ptr_vec_t& BackProp(const tensor_ptr_vec_t& outputsGradient) override;

        virtual const vector<LayerBase*>& ModelInputLayers() const override { return m_ModelInputLayers; }
        virtual const vector<LayerBase*>& ModelOutputLayers() const override { return m_ModelOutputLayers; }
        virtual const vector<LayerBase*>& Layers() const override { return m_Order; }

    protected:
        virtual LayerBase* LinkImpl(const vector<LayerBase*>& inputLayers) override;
        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnInit(bool initValues = true) override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
        virtual void OnLinkOutput(LayerBase* outputLayer) override;

	private:
		Flow() {}

        void ProcessLayer(LayerBase* layer, vector<LayerBase*>& visited);

        vector<Tensor*> m_InputsGradient;
        tensor_ptr_vec_t m_Outputs;
        vector<Shape> m_OutputsShapes;
        
        vector<LayerBase*> m_ModelInputLayers;
        vector<LayerBase*> m_ModelOutputLayers;

        vector<LayerBase*> m_Order;
        vector<LayerBase*> m_ReversedOrder;
	};
}
