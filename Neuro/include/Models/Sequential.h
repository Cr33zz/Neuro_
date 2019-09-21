#pragma once

#include <string>

#include "Models/ModelBase.h"

namespace Neuro
{
	using namespace std;

    class Sequential : public ModelBase
    {
	public:
		Sequential(const string& name = "", int seed = 0);
        ~Sequential();

        virtual const vector<Shape>& InputShapes() const override { return m_Layers[0]->InputShapes(); }
        virtual const vector<Tensor*>& InputsGradient() override { return m_Layers[0]->InputsGradient(); }
        virtual const tensor_ptr_vec_t& Outputs() const override { return m_Layers.back()->Outputs(); }
        virtual const vector<Shape>& OutputShapes() const override { return m_Layers.back()->OutputShapes(); }
        virtual const vector<LayerBase*>& InputLayers() const override { return m_Layers[0]->InputLayers(); }
        virtual const vector<LayerBase*>& OutputLayers() const override { return m_Layers.back()->OutputLayers(); }

        virtual const tensor_ptr_vec_t& FeedForward(const const_tensor_ptr_vec_t& inputs, bool training) override;
        virtual const tensor_ptr_vec_t& BackProp(const tensor_ptr_vec_t& outputsGradient) override;

        virtual int InputOffset(const LayerBase* inputLayer) const override;

        virtual const vector<LayerBase*>& ModelInputLayers() const override { return m_ModelInputLayers; }
        virtual const vector<LayerBase*>& ModelOutputLayers() const override { return m_ModelOutputLayers; }
        virtual const vector<LayerBase*>& Layers() const override { return m_Layers; }

        LayerBase* Layer(int i) { return m_Layers[i]; }
        LayerBase* LastLayer() const { return m_Layers.back(); }
        void AddLayer(LayerBase* layer);

    protected:
        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnLink(LayerBase* layer, bool input) override;

	private:
        Sequential(int) {}

        vector<LayerBase*> m_Layers;
        vector<LayerBase*> m_ModelInputLayers;
        vector<LayerBase*> m_ModelOutputLayers;
	};
}
