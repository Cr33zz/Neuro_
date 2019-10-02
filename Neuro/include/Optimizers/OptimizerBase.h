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

        virtual Operation* Minimize(TensorLike* loss) = 0;
        //virtual Operation* Maximize(NodeBase* loss) = 0;

        static vector<Variable*> ComputeGradients(TensorLike* loss);

	protected:
		float m_Iteration = 0;
	};
}
