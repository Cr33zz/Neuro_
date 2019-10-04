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
    void SGD::MinimizationOperation::ComputeInternal()
    {
        auto vars = Graph::Default()->ComputeGradients(m_InputNodes);

        for (auto v : vars)
            Tensor::ActiveOp()->SgdStep(v->Output(), v->OutputGrad(), (float)v->Output().Batch(), m_Owner->m_LearningRate);
    }
}
