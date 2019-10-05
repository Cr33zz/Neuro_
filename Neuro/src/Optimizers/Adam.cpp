#include <sstream>
#include <iomanip>

#include "Optimizers/Adam.h"
#include "Tensors/Tensor.h"
#include "Tensors/TensorOpCpu.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{    
	//////////////////////////////////////////////////////////////////////////
    Adam::Adam(float lr, float beta1, float beta2)
	{
		m_LearningRate = lr;
        m_Beta1 = beta1;
        m_Beta2 = beta2;
	}

    //////////////////////////////////////////////////////////////////////////
    OptimizerBase* Adam::Clone() const
    {
        return new Adam(*this);
    }

    //////////////////////////////////////////////////////////////////////////
	string Adam::ToString()
	{
		stringstream ss;
		ss << setprecision(5) << "Adam(lr=" << m_LearningRate << ", beta1=" << m_Beta1 << ", beta2=" << m_Beta2 << ")";
		return ss.str();
	}

	//////////////////////////////////////////////////////////////////////////
	const char* Adam::ClassName() const
	{
		return "Adam";
	}

    //////////////////////////////////////////////////////////////////////////
    void Adam::MinimizationOperation::ComputeInternal()
    {
        auto vars = Graph::Default()->ComputeGradients(m_InputNodes);
        ++m_Owner->m_Iteration;

        float batchSize = (float)m_Inputs[0]->Batch(); // assuming all inputs have the same batch size

        if (m_MGradients.size() != vars.size())
        {
            assert(m_MGradients.empty() && m_VGradients.empty());

            for (uint32_t i = 0; i < vars.size(); ++i)
            {
                m_MGradients.push_back(Tensor(vars[i]->Output().GetShape()));
                m_VGradients.push_back(Tensor(vars[i]->Output().GetShape()));
            }
        }

        float learningRate = m_Owner->m_LearningRate * (float)::sqrt(1.0 - ::pow(m_Owner->m_Beta2, m_Owner->m_Iteration)) / (1.0f - (float)::pow(m_Owner->m_Beta1, m_Owner->m_Iteration));

        for (auto i = 0; i < vars.size(); ++i)
        {
            auto& parameter = vars[i]->Output();
            auto& gradient = vars[i]->OutputGrad();
            auto& mGrad = m_MGradients[i];
            auto& vGrad = m_VGradients[i];

            Tensor::ActiveOp()->AdamStep(parameter, gradient, mGrad, vGrad, batchSize, learningRate, m_Owner->m_Beta1, m_Owner->m_Beta2, m_Owner->m_Epsilon);
        }
    }
}
