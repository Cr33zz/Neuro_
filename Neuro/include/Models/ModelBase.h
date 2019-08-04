#pragma once

#include <string>
#include <vector>

#include "Types.h"
#include "ParametersAndGradients.h"

namespace Neuro
{
	using namespace std;

	class LayerBase;
	class Tensor;

    class ModelBase
    {
	public:
        virtual ModelBase* Clone() const = 0;
		virtual void FeedForward(const tensor_ptr_vec_t& inputs) = 0;
        virtual void BackProp(vector<Tensor>& deltas) = 0;
        virtual void Optimize() { }
        virtual const vector<LayerBase*>& GetLayers() const = 0;
		virtual tensor_ptr_vec_t GetOutputs() const = 0;
        virtual const vector<LayerBase*>& GetOutputLayers() const = 0;
        virtual int GetOutputLayersCount() const = 0;
        virtual string Summary() const { return ""; }
        virtual void SaveStateXml(string filename) const { }
        virtual void LoadStateXml(string filename) { }

        const LayerBase* GetLayer(const string& name) const;
        vector<ParametersAndGradients> GetParametersAndGradients();
	};
}
