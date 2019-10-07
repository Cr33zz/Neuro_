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

        void AddLayer(LayerBase* layer);

    protected:
        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;

	private:
        Sequential(int) {}
	};
}
