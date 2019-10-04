#pragma once

#include <vector>
#include <string>
#include "ComputationalGraph/Operation.h"

namespace Neuro
{
	using namespace std;

    class Variable;

    class OptimizerBase
    {
	public:
        virtual OptimizerBase* Clone() const = 0;
		virtual string ToString() = 0;
		virtual const char* ClassName() const = 0;

        int Iteration() const { return (int)m_Iteration; }

        virtual Operation* Minimize(const vector<TensorLike*>& losses) = 0;
        //virtual Operation* Maximize(const vector<TensorLike*>& losses) = 0;

	protected:
		float m_Iteration = 0;
	};
}
