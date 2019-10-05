#pragma once

#include <vector>
#include <unordered_set>

namespace Neuro
{
    using namespace std;

    class TensorLike;
    class Placeholder;
    class Operation;
    class Variable;

    class Graph
    {
    public:
        Graph();

        //void SetAsDefault() { s_Default = this; }
        static Graph* Default();

        void AddVariable(Variable* v);
        void AddPlaceholder(Placeholder* p);
        void AddOperation(Operation* op);

        const vector<Variable*>& Variables() const { return m_Variables; }
        const vector<Placeholder*>& Placeholders() const { return m_Placeholders; }
        const vector<Operation*>& Operations() const { return m_Operations; }

        void InitVariables();
        void IncrementStep();
        uint32_t CurrentStep() const { return m_CurrentStep; }

        // Builds nodes visitation order for forward pass
        vector<TensorLike*> BuildForwardOrder(const vector<TensorLike*>& endNodes);
        // Builds nodes visitation order for backward/gradients computation pass
        vector<TensorLike*> BuildBackwardOrder(const vector<TensorLike*>& endNodes, const vector<Variable*>& params = {}, bool inludeEndNodes = true);

        vector<Variable*> ComputeGradients(const vector<TensorLike*>& losses, const vector<Variable*>& params);
        vector<Variable*> ComputeGradientsInOrder(const vector<TensorLike*>& order, const vector<Variable*>& params);

        void DebugLog();

    private:
        void ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes, unordered_set<TensorLike*>& visited);
        void ProcessBackwardNode(TensorLike* node, vector<TensorLike*>& nodes, const vector<Variable*>& params, bool ignoreConsumersCheck, unordered_set<TensorLike*>& visited, unordered_set<TensorLike*>& visitedParams, const unordered_set<TensorLike*>& required);

        vector<Placeholder*> m_Placeholders;
        vector<Operation*> m_Operations;
        vector<Variable*> m_Variables;
        vector<TensorLike*> m_Nodes;
        bool m_VariablesInitialized = false;
        uint32_t m_CurrentStep = 0;

        static Graph* s_Default;
    };
}
