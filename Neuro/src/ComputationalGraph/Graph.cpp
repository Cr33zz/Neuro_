#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Operation.h"
#include "Debug.h"
#include "Tools.h"

namespace Neuro
{
    Graph* Graph::s_Default = nullptr;

    //////////////////////////////////////////////////////////////////////////
    Graph::Graph()
    {        
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Graph* Graph::Default()
    {
        if (!s_Default)
            s_Default = new Graph();
        return s_Default;
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::AddVariable(Variable* v)
    {
        m_Variables.push_back(v);
        m_Nodes.push_back(v);
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::AddPlaceholder(Placeholder* p)
    {
        m_Placeholders.push_back(p);
        m_Nodes.push_back(p);
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::AddOperation(Operation* op)
    {
        m_Operations.push_back(op);
        m_Nodes.push_back(op);
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::InitVariables()
    {
        if (m_VariablesInitialized)
            return;

        for (auto var : m_Variables)
            var->Init();

        m_VariablesInitialized = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Graph::BuildForwardOrder(const vector<TensorLike*>& endNodes)
    {
        vector<TensorLike*> result;
        for (auto node : endNodes)
            ProcessForwardNode(node, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::DebugLog()
    {
#ifdef DEBUG_LOG_ENABLED
        for (auto node : m_Nodes)
        {
            if (Debug::ShouldLogOutput(node->Name()))
                node->Output().DebugDumpValues(Replace(node->Name() + "_step" + to_string(Debug::GetStep()) + ".log", "/", "_"));
            if (Debug::ShouldLogGrad(node->Name()))
                node->OutputGrad().DebugDumpValues(Replace(node->Name() + "_grad_step" + to_string(Debug::GetStep()) + ".log", "/", "_"));
        }
#endif
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes)
    {
        for (auto inputNode : node->m_InputNodes)
            ProcessForwardNode(inputNode, nodes);
        nodes.push_back(node);
    }
}
