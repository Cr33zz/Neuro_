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

        virtual const Shape& InputShape() const override { return m_Layers[0]->InputShape(); }
        virtual const tensor_ptr_vec_t& Outputs() const override { return m_Layers.back()->Outputs(); }
        virtual const vector<Shape>& OutputShapes() const override { return m_Layers.back()->OutputShapes(); }
        virtual const vector<LayerBase*>& InputLayers() const override { return m_Layers[0]->InputLayers(); }
        virtual const vector<LayerBase*>& OutputLayers() const override { return m_Layers.back()->OutputLayers(); }

        virtual const vector<LayerBase*>& ModelInputLayers() const override { return m_ModelInputLayers; }
        virtual const vector<LayerBase*>& ModelOutputLayers() const override { return m_ModelOutputLayers; }
        virtual const vector<LayerBase*>& Layers() const override { return m_Layers; }

        LayerBase* Layer(int i) { return m_Layers[i]; }
        LayerBase* LastLayer() const { return m_Layers.back(); }
        void AddLayer(LayerBase* layer);

    protected:
        virtual LayerBase* LinkImpl(const vector<LayerBase*>& inputLayers) override;
        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
        virtual void OnLinkOutput(LayerBase* outputLayer) override;

	private:
        Sequential(int) {}

        vector<LayerBase*> m_Layers;
        vector<LayerBase*> m_ModelInputLayers;
        vector<LayerBase*> m_ModelOutputLayers;
	};
}
