#pragma once

#include <vector>
#include <string>
#include "ComputationalGraph/Operation.h"

namespace Neuro
{
	using namespace std;

    class Variable;

    class NEURO_DLL_EXPORT OptimizerBase
    {
	public:
        virtual OptimizerBase* Clone() const = 0;
		virtual string ToString() = 0;
		virtual const char* ClassName() const = 0;

        virtual Operation* Minimize(const vector<TensorLike*>& losses, const vector<Variable*>& vars = {}, Variable* globalStep = nullptr) = 0;
        //virtual Operation* Maximize(const vector<TensorLike*>& losses) = 0;
	};
}
