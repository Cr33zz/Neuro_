#include <sstream>
#include <iomanip>

#include "Optimizers/SGD.h"
#include "Tensors/TensorOpCpu.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	SGD::SGD(float lr)
	{
		m_LearningRate = lr;
	}

    //////////////////////////////////////////////////////////////////////////
    OptimizerBase* SGD::Clone() const
    {
        return new SGD(*this);
    }

    //////////////////////////////////////////////////////////////////////////
	string SGD::ToString()
	{
		stringstream ss;
		ss << setprecision(5) << "SGD(lr=" << m_LearningRate << ")";
		return ss.str();
	}

	//////////////////////////////////////////////////////////////////////////
	const char* SGD::ClassName() const
	{
		return "SGD";
	}

    //////////////////////////////////////////////////////////////////////////
    SGD::MinimizationOperation::MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, float lr)
        : Operation(losses, "sgd_minimize"), m_Vars(vars), m_LearningRate(lr)
    {
        m_Order = Graph::Default()->BuildBackwardOrder(losses, m_NodesAffectingLosses, vars);
    }

    //////////////////////////////////////////////////////////////////////////
    void SGD::MinimizationOperation::ComputeInternal()
    {
        m_InputsManuallyConsumed = true; // loss outputs will be completely obliterated after gradients computation
        auto vars = Graph::Default()->ComputeGradientsInOrder(m_Order, m_InputNodes, m_NodesAffectingLosses, m_Vars);

        for (auto v : vars)
            Tensor::ActiveOp()->SgdStep(v->Output(), v->OutputGrad(), /*batchSize, */m_LearningRate);
    }
}
