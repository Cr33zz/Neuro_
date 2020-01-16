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
    class Constant;

    class Graph
    {
    public:
        Graph();

        //void SetAsDefault() { s_Default = this; }
        static Graph* Default();

        void AddVariable(Variable* v);
        void AddConstant(Constant* c);
        void AddPlaceholder(Placeholder* p);
        void AddOperation(Operation* op);

        void Clear();

        const vector<Variable*>& Variables() const { return m_Variables; }
        const vector<Constant*>& Constants() const { return m_Constants; }
        const vector<Placeholder*>& Placeholders() const { return m_Placeholders; }
        const vector<Operation*>& Operations() const { return m_Operations; }

        void InitVariables();
        void IncrementStep();
        uint32_t CurrentStep() const { return m_CurrentStep; }

        // Builds nodes visitation order for forward pass, returns true when order contains training operation
        bool BuildForwardOrder(const vector<TensorLike*>& endNodes, vector<TensorLike*>& order);
        // Builds nodes visitation order for backward/gradients computation pass
        vector<TensorLike*> BuildBackwardOrder(const vector<TensorLike*>& endNodes, unordered_set<TensorLike*>& nodesAffectingEndNodes, const vector<Variable*>& params = {});

        vector<Variable*> ComputeGradients(const vector<TensorLike*>& losses, const vector<Variable*>& params);
        vector<Variable*> ComputeGradientsInOrder(const vector<TensorLike*>& order, const vector<TensorLike*>& losses, const unordered_set<TensorLike*> nodesAffectingLosses, const vector<Variable*>& params);

        TensorLike* GetNode(const string& name);
        void DebugLog();

    private:
        void ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes, unordered_set<TensorLike*>& visited, bool& is_training);
        void ProcessBackwardNode(TensorLike* node, vector<TensorLike*>& nodes, const vector<Variable*>& params, bool ignoreConsumersCheck, unordered_set<TensorLike*>& visited, unordered_set<TensorLike*>& visitedParams, const unordered_set<TensorLike*>& required);

        vector<Placeholder*> m_Placeholders;
        vector<Operation*> m_Operations;
        vector<Variable*> m_Variables;
        vector<Constant*> m_Constants;
        vector<TensorLike*> m_Nodes;
        uint32_t m_CurrentStep = 0;

        static Graph* s_Default;
    };
}
