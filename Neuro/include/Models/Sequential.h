#pragma once

#include <string>

#include "Models/Flow.h"

namespace Neuro
{
	using namespace std;

    class NEURO_DLL_EXPORT Sequential : public Flow
    {
	public:
		Sequential(const string& name = "", int seed = 0);
        ~Sequential();

        void AddLayer(LayerBase* layer);

    protected:
        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnClone(const LayerBase& source) override;

        virtual void Build(const vector<Shape>& inputShapes) override;

	private:
        //Sequential(int) {}
	};
}
