#include <unordered_set>
#include <list>

#include "Optimizers/OptimizerBase.h"
#include "ComputationalGraph/Variable.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    vector<Variable*> OptimizerBase::ComputeGradients(TensorLike* lossNode)
    {
        vector<Variable*> variables;
        lossNode->m_OutputGrad.Resize(lossNode->m_Output.GetShape());
        lossNode->m_OutputGrad.FillWithValue(1);

        unordered_set<TensorLike*> visited;
        list<TensorLike*> queue;

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
                    assert(outputGrad.Length());
                    auto& inputsGrad = static_cast<Operation*>(consumer)->ComputeGradient(outputGrad);

                    if (inputsGrad.size() == 1)
                    {
                        assert(inputsGrad[0]->Length());
                        nodeGrad.Add(*inputsGrad[0], nodeGrad);
                    }
                    else
                    {
                        auto nodeIndexInConsumerInputs = distance(consumer->m_InputNodes.begin(), find(consumer->m_InputNodes.begin(), consumer->m_InputNodes.end(), node));
                        auto lossGradWrtNode = inputsGrad[nodeIndexInConsumerInputs];
                        assert(lossGradWrtNode->Length());
                        nodeGrad.Add(*lossGradWrtNode, nodeGrad);
                    }
                }
            }

            for (auto inputNode : node->m_InputNodes)
            {
                if (visited.find(inputNode) != visited.end())
                    continue;

                //do not enqueue nodes where there are not visited consumers (that can happen often for same placeholders used in mutiple operations
                bool allConsumersVisited = true;
                for (auto consumer : inputNode->m_Consumers)
                {
                    if (visited.find(consumer) == visited.end())
                    {
                        allConsumersVisited = false;
                        break;
                    }
                }
                
                if (!allConsumersVisited)
                    continue; // we will add this input node as soon as its last consumer was visited

                visited.insert(inputNode);
                queue.push_back(inputNode);
            }
        }

        return variables;
    }
}
