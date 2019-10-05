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
    SGD::MinimizationOperation::MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, SGD* owner)
        : Operation(losses, "sgd_minimize"), m_Owner(owner), m_Vars(vars)
    {
        m_Order = Graph::Default()->BuildBackwardOrder(losses, vars, false);
    }

    //////////////////////////////////////////////////////////////////////////
    void SGD::MinimizationOperation::ComputeInternal()
    {
        auto vars = Graph::Default()->ComputeGradientsInOrder(m_Order, m_Vars);
        float batchSize = (float)m_Inputs[0]->Batch(); // assuming all inputs have the same batch size

        for (auto v : vars)
            Tensor::ActiveOp()->SgdStep(v->Output(), v->OutputGrad(), batchSize, m_Owner->m_LearningRate);
    }
}
