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
		virtual ModelBase* Clone() const override;

		virtual void FeedForward(const vector<const Tensor*>& inputs) override;
		virtual void BackProp(vector<Tensor>& deltas) override;
		virtual vector<const Tensor*> GetOutputs() const override;
		virtual const vector<LayerBase*>& GetOutputLayers() const override;
		virtual int GetOutputLayersCount() const override;
        virtual void Optimize();
		virtual const vector<LayerBase*>& GetLayers() const override;
		virtual string Summary() const override;

	private:
		// For cloning purposes
		Flow();
        void ProcessLayer(LayerBase* layer, vector<LayerBase*>& visited);
        
        vector<LayerBase*> InputLayers;
        vector<LayerBase*> OutputLayers;

        vector<LayerBase*> Order;
        vector<LayerBase*> ReversedOrder;
	};
}
