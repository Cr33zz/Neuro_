#include <unordered_set>
#include <list>

#include "ComputationalGraph/Optimizer.h"
#include "ComputationalGraph/Variable.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    vector<Variable*> Optimizer::ComputeGradients(NodeBase* lossNode)
    {
        vector<Variable*> variables;
        lossNode->m_OutputGrad.Resize(lossNode->m_Output.GetShape());
        lossNode->m_OutputGrad.FillWithValue(1);

        unordered_set<NodeBase*> visited;
        list<NodeBase*> queue;

        visited.insert(lossNode);
        queue.push_back(lossNode);

        while (!queue.empty())
        {
            auto node = queue.front();
            queue.pop_front();

            if (Variable* v = dynamic_cast<Variable*>(node))
                variables.push_back(v);

            if (node != lossNode)
            {
                auto& nodeGrad = node->m_OutputGrad;
                nodeGrad.Resize(node->m_Output.GetShape());
                nodeGrad.Zero(); // reset gradient

                for (auto consumer : node->m_Consumers)
                {
                    auto& outputGrad = consumer->m_OutputGrad;
                    auto& inputsGrad = static_cast<Operation*>(consumer)->ComputeGradient(outputGrad);

                    if (inputsGrad.size() == 1)
                    {
                        nodeGrad.Add(*inputsGrad[0], nodeGrad);
                    }
                    else
                    {
                        auto nodeIndexInConsumerInputs = distance(consumer->m_InputNodes.begin(), find(consumer->m_InputNodes.begin(), consumer->m_InputNodes.end(), node));
                        auto lossGradWrtNode = inputsGrad[nodeIndexInConsumerInputs];
                        nodeGrad.Add(*lossGradWrtNode, nodeGrad);
                    }
                }
            }

            for (auto inputNode : node->m_InputNodes)
            {
                if (visited.find(inputNode) != visited.end())
                    continue;

                visited.insert(inputNode);
                queue.push_back(inputNode);
            }
        }

        return variables;
    }

    //////////////////////////////////////////////////////////////////////////
    void _SGDOptimizer::MinimizationOperation::ComputeInternal()
    {
        auto vars = Optimizer::ComputeGradients(m_InputNodes[0]);

        for (auto v : vars)
            v->Output().Sub(v->OutputGrad().Mul(m_LearningRate), v->Output());
    }

}