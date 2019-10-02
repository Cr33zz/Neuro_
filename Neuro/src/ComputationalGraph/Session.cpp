#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    Session* Session::Default = new Session();

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor*> Session::Run(const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds)
    {
        for(auto node : BuildForwardGraph(fetches))
        {
            if (Placeholder* p = dynamic_cast<Placeholder*>(node))
                node->m_Output = *feeds.find(p)->second;
            else if (node->IsOp())
            {
                Operation* op = static_cast<Operation*>(node);
                auto inputs = op->GatherInputs();
                op->Compute(inputs);
            }
        }
        vector<Tensor*> result(fetches.size());
        for (size_t i = 0; i < fetches.size(); ++i)
            result[i] = fetches[i]->OutputPtr();
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    std::vector<TensorLike*> Session::BuildForwardGraph(const vector<TensorLike*>& endNodes)
    {
        vector<TensorLike*> result;
        for (auto node : endNodes)
            ProcessForwardNode(node, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Session::ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes)
    {
        if (node->IsOp())
        {
            Operation* op = static_cast<Operation*>(node);
            for (auto inputNode : op->m_InputNodes)
                ProcessForwardNode(inputNode, nodes);
        }
        nodes.push_back(node);
    }
}
