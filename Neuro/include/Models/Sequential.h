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
        virtual ModelBase* Clone() const override;
        virtual void FeedForward(const tensor_ptr_vec_t& inputs) override;
        virtual void BackProp(vector<Tensor>& deltas) override;
        virtual tensor_ptr_vec_t GetOutputs() const override;
        virtual const vector<LayerBase*>& GetOutputLayers() const override;
        virtual int GetOutputLayersCount() const override;
        virtual const vector<LayerBase*>& GetLayers() const override;
        virtual string Summary() const override;
        virtual void SaveStateXml(string filename) const override;
        virtual void LoadStateXml(string filename) override;

		LayerBase* GetLayer(int i);
        LayerBase* GetLastLayer() const;
        int LayersCount() const;
        void AddLayer(LayerBase* layer);

	private:
        vector<LayerBase*> m_Layers;
		vector<LayerBase*> m_OutputLayers;
	};
}
