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
    class NEURO_DLL_EXPORT Flow : public ModelBase
    {
	public:
        Flow(const vector<TensorLike*>& inputs, const vector<TensorLike*>& outputs, const string& name = "", int seed = 0);
        ~Flow();

    protected:
        Flow(const string& constructorName, const string& name = "", int seed = 0);

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;

	private:
		Flow() {}
	};
}
