#pragma once

#include <map>
#include <vector>

#include "Models/ModelBase.h"

namespace Neuro
{
	using namespace std;

    class Flow : public ModelBase
    {
	public:
        Flow(const vector<LayerBase*>& inputLayers, const vector<LayerBase*>& outputLayers, const string& name = "", int seed = 0);
        ~Flow();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;

        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(vector<Tensor>& outputGradients) override;

		virtual const vector<LayerBase*>& GetOutputLayers() const override;
		virtual uint32_t GetOutputLayersCount() const override;
		virtual const vector<LayerBase*>& GetLayers() const override;

	private:
		Flow() {}

        void ProcessLayer(LayerBase* layer, vector<LayerBase*>& visited);
        
        vector<LayerBase*> m_ModelInputLayers;
        vector<LayerBase*> m_ModelOutputLayers;

        vector<LayerBase*> m_Order;
        vector<LayerBase*> m_ReversedOrder;
	};
}
