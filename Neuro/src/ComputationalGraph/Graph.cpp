#include <fstream>

#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Operation.h"
#include "Debug.h"
#include "Tools.h"
#include "Memory/MemoryManager.h"

//#define ENABLE_GRAPH_LOGS

#ifdef ENABLE_GRAPH_LOGS
#include <windows.h>
#include <debugapi.h>
#include "Tools.h"
#define GRAPH_DEBUG_INFO(...) do { static char timeBuffer[128]; SYSTEMTIME sysTime; GetLocalTime(&sysTime); sprintf(timeBuffer, "%02d:%02d:%02d.%03d - ", sysTime.wHour, sysTime.wMinute, sysTime.wSecond, sysTime.wMilliseconds); OutputDebugString(timeBuffer); static char buffer[1024]; sprintf(buffer, __VA_ARGS__); OutputDebugString(buffer); } while (0)
#else
#define GRAPH_DEBUG_INFO(...) {}
#endif

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
        for (auto var : m_Variables)
            var->Initialize();
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::IncrementStep()
    {
        ++m_CurrentStep;
    }

    //////////////////////////////////////////////////////////////////////////
    bool Graph::BuildForwardOrder(const vector<TensorLike*>& endNodes, vector<TensorLike*>& order)
    {
        bool isTraining = false;
        unordered_set<TensorLike*> visited;

        for (auto node : endNodes)
            ProcessForwardNode(node, order, visited, isTraining);

        return isTraining;
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes, unordered_set<TensorLike*>& visited, bool& is_training)
    {
        if (visited.find(node) != visited.end())
            return;

        for (auto inputNode : node->m_InputNodes)
            ProcessForwardNode(inputNode, nodes, visited, is_training);

        is_training |= (node->IsOp() && static_cast<Operation*>(node)->IsTrainingOp());

        visited.insert(node);
        nodes.push_back(node);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Graph::BuildBackwardOrder(const vector<TensorLike*>& endNodes, unordered_set<TensorLike*>& nodesAffectingEndNodes, const vector<Variable*>& params)
    {
        for (auto param : params)
            NEURO_ASSERT(param->CareAboutGradient(), "Parameter '" << param->Name() << "' doesn't care about gradient.");

        // we need to figure out which nodes were required to calculate end nodes
        // later on when check if all consumers were visited we will additionally check if
        // any particular consumer is required, otherwise it's inputs' gradients are not important 
        // (and won't be computed anyway)
        vector<TensorLike*> tempNodesAffectingEndNodes;
        BuildForwardOrder(endNodes, tempNodesAffectingEndNodes);

        // build hash set for fast lookup
        nodesAffectingEndNodes.clear();
        nodesAffectingEndNodes.insert(tempNodesAffectingEndNodes.begin(), tempNodesAffectingEndNodes.end());

        vector<TensorLike*> result;
        unordered_set<TensorLike*> visited;
        unordered_set<TensorLike*> visitedParams;

        for (auto node : endNodes)
            ProcessBackwardNode(node, result, params, false, visited, visitedParams, nodesAffectingEndNodes);

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::ProcessBackwardNode(TensorLike* node, vector<TensorLike*>& nodes, const vector<Variable*>& params, bool ignoreConsumersCheck, unordered_set<TensorLike*>& visited, unordered_set<TensorLike*>& visitedParams, const unordered_set<TensorLike*>& required)
    {
        bool allConsumersVisited = true;

        if (!ignoreConsumersCheck)
        {
            // only include node when all consumers are already visited
            // we have to make sure that all consumers' input gradients are computed in order to properly average node's output gradient
            for (auto consumer : node->m_Consumers)
            {
                assert(consumer->IsOp());

                // we don't care about consumers that didn't contribute to losses
                if (required.find(consumer) == required.end())
                    continue;

                if (visited.find(consumer) == visited.end())
                {
                    allConsumersVisited = false;
                    break;
                }
            }
        }

        if (!allConsumersVisited)
            return;

        visited.insert(node);
        nodes.push_back(node);

        // we can stop back propagation as soon as we have visited all desired parameters
        if (!params.empty() && node->IsVar() && find(params.begin(), params.end(), node) != params.end())
        {
            visitedParams.insert(node);
            if (visitedParams.size() == params.size())
                return;
        }

        for (auto inputNode : node->m_InputNodes)
        {
            if (visited.find(inputNode) != visited.end())
                continue;

            ProcessBackwardNode(inputNode, nodes, params, false, visited, visitedParams, required);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Variable*> Graph::ComputeGradients(const vector<TensorLike*>& losses, const vector<Variable*>& params)
    {
        unordered_set<TensorLike*> nodesAffectingLosses;
        auto order = BuildBackwardOrder(losses, nodesAffectingLosses, params);
        return ComputeGradientsInOrder(order, losses, nodesAffectingLosses, params);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Variable*> Graph::ComputeGradientsInOrder(const vector<TensorLike*>& order, const vector<TensorLike*>& losses, const unordered_set<TensorLike*> nodesAffectingLosses, const vector<Variable*>& params)
    {

        DeviceMemoryManager::Default().ForceMemoryStreamSync();

        const size_t PREFETCH_STEPS = 1;
        vector<Variable*> variables;

        /// remove all node which don't care about gradient. it has to be done at runtime since variables can be switched between trainable and non-trainable state
        /// between consecutive session runs
        auto newOrder = order;
        newOrder.erase(remove_if(newOrder.begin(), newOrder.end(), [](const TensorLike* node) { return !node->CareAboutGradient(); }), newOrder.end());

        size_t lastPrefetched = 0;
        
        for (size_t n = 0; n < newOrder.size(); ++n)
        {
            for (size_t p = lastPrefetched + 1; p <= n + PREFETCH_STEPS; ++p)
            {
                if (p >= newOrder.size())
                    break;

                auto node = newOrder[p];
                NVTXProfile nvtxProf((string("Preload ") + node->Name()).c_str(), 0xFF5BB8FF);
                GRAPH_DEBUG_INFO("##Graph: Preloading '%s'...\n", node->Name().c_str());
                node->PreloadForGradient();
            }
            lastPrefetched = n + PREFETCH_STEPS;

            auto node = newOrder[n];
            GRAPH_DEBUG_INFO("##Graph: Computing gradient '%s'... (care about grad: %d)\n", node->Name().c_str(), node->CareAboutGradient() ? 1 : 0);

            if (node->CareAboutGradient())
            {
                NVTXProfile nvtxProf((string("Output grad for ") + node->Name()).c_str(), 0xFF4242FF);
                if (node->IsVar())
                {
                    Variable* var = static_cast<Variable*>(node);
                    if (var->Trainable() && (params.empty() || find(params.begin(), params.end(), node) != params.end()))
                        variables.push_back(var);
                }

                auto& nodeOutputGrad = node->m_OutputGrad;
                nodeOutputGrad.Resize(node->m_Output.GetShape());
                if (nodeOutputGrad.TryDeviceAllocate())
                    nodeOutputGrad.OverrideDevice();
                nodeOutputGrad.Zero(); // reset gradient

                if (find(losses.begin(), losses.end(), node) != losses.end())
                {
                    // gradient of loss w.r.t to loss is 1
                    nodeOutputGrad.One();
                }
                else
                {
                    for (auto consumer : node->m_Consumers)
                    {
                        assert(consumer->IsOp());
                        Operation* consumerOp = static_cast<Operation*>(consumer);

                        // ignore consumer when it didn't affect loss. one example of such consumers might be accuracy operation
                        if (nodesAffectingLosses.find(consumer) == nodesAffectingLosses.end())
                            continue;

                        auto& inputsGrad = consumerOp->InputsGrads();

                        if (inputsGrad.size() == 1)
                        {
                            assert(inputsGrad[0].Length());
                            nodeOutputGrad.Add(inputsGrad[0], nodeOutputGrad);
                        }
                        else
                        {
                            auto nodeIndexInConsumerInputs = distance(consumer->m_InputNodes.begin(), find(consumer->m_InputNodes.begin(), consumer->m_InputNodes.end(), node));
                            auto& lossGradWrtNode = inputsGrad[nodeIndexInConsumerInputs];
                            assert(lossGradWrtNode.Length());
                            nodeOutputGrad.Add(lossGradWrtNode, nodeOutputGrad);
                        }
                    }
                }

                Operation* opNode = node->IsOp() ? static_cast<Operation*>(node) : nullptr;
                
                if (opNode)
                {
                    NVTXProfile nvtxProf((string("Compute grad ") + node->Name()).c_str(), 0xFF4242FF);
                    opNode->ComputeGradient(nodeOutputGrad);

                    if (Debug::ShouldLogGrad(node->Name()))
                    {
                        nodeOutputGrad.DebugDumpValues(node->Name() + "_output0_grad_step" + to_string(Debug::GetStep()) + ".log");
                        for (size_t i = 0; i < opNode->InputsGrads().size(); ++i)
                        {
                            if (opNode->InputNodes()[i]->CareAboutGradient())
                                opNode->InputsGrads()[i].DebugDumpValues(node->Name() + "_input" + to_string(i) + "_grad_step" + to_string(Debug::GetStep()) + ".log");
                            else
                            {
                                ofstream s(node->Name() + "_input" + to_string(i) + "_grad_step" + to_string(Debug::GetStep()) + ".log");
                                s << "doesn't care about gradient";
                                s.close();
                            }
                        }
                    }

                    node->Output().DecRef(); // output is no longer needed, we've already used it to compute input gradients
                    node->OutputGrad().ReleaseData(); // output grad is no longer needed, we've already used it to compute input gradients
                }
                else
                {
                    if (Debug::ShouldLogGrad(node->Name()))
                        nodeOutputGrad.DebugDumpValues(node->Name() + "_grad_step" + to_string(Debug::GetStep()) + ".log");
                }
            }

            // all consumers contributing to this node's output grad can be notified so they can release their corresponding input gradient
            for (auto consumerNode : node->m_Consumers)
                consumerNode->InputGradConsumed(node);
        }

        return variables;
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* Graph::GetNode(const string& name)
    {
        for (auto node : m_Nodes)
        {
            if (node->Name() == name)
                return node;
        }
        return nullptr;
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::DebugLog()
    {
        for (auto node : m_Variables)
        {
            if (Debug::ShouldLogGrad(node->Name()))
                node->OutputGrad().DebugDumpValues(node->Name() + "_grad_step" + to_string(m_CurrentStep) + ".log");
        }
    }
}
