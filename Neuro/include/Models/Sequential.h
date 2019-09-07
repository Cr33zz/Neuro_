#pragma once

#include <string>

#include "Models/ModelBase.h"

namespace Neuro
{
	using namespace std;

    class Sequential : public ModelBase
    {
	public:
		Sequential();
        virtual ~Sequential();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;

        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputGradients) override;
        
        virtual const vector<LayerBase*>& GetOutputLayers() const override;
        virtual uint32_t GetOutputLayersCount() const override;
        virtual const vector<LayerBase*>& GetLayers() const override;
        virtual void SaveStateXml(string filename) const override;
        virtual void LoadStateXml(string filename) override;

		LayerBase* GetLayer(int i);
        LayerBase* LastLayer() const;
        int LayersCount() const;
        void AddLayer(LayerBase* layer);

	private:
        vector<LayerBase*> m_Layers;
		vector<LayerBase*> m_OutputLayers;
	};
}
