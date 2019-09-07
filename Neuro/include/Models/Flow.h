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
        Flow(const vector<LayerBase*>& inputLayers, const vector<LayerBase*>& outputLayers);
        virtual ~Flow();

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
        
        vector<LayerBase*> m_InputLayers;
        vector<LayerBase*> m_OutputLayers;

        vector<LayerBase*> m_Order;
        vector<LayerBase*> m_ReversedOrder;
	};
}
