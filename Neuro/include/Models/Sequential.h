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

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;

        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputsGradient) override;
        
        virtual const vector<LayerBase*>& OutputLayers() const override;
        virtual uint32_t OutputLayersCount() const override;
        virtual const vector<LayerBase*>& Layers() const override;

		LayerBase* Layer(int i);
        LayerBase* LastLayer() const;
        int LayersCount() const;
        void AddLayer(LayerBase* layer);

	private:
        Sequential(int) {}

        vector<LayerBase*> m_Layers;
		vector<LayerBase*> m_OutputLayers;
	};
}
