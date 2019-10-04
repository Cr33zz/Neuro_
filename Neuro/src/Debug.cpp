#include "Debug.h"

namespace Neuro
{
    int Debug::g_Step = 0;
    unordered_set<string> Debug::g_LogOutputs;
    unordered_set<string> Debug::g_LogGrads;
    bool Debug::g_LogAllOutputs = false;
    bool Debug::g_LogAllGrads = false;

    //////////////////////////////////////////////////////////////////////////
    void Debug::Step()
    {
        ++g_Step;
    }

    //////////////////////////////////////////////////////////////////////////
    int Debug::GetStep()
    {
        return g_Step;
    }

    //////////////////////////////////////////////////////////////////////////
    void Debug::LogOutput(const string& name, bool enable)
    {
        if (enable)
            g_LogOutputs.insert(name);
        else
            g_LogOutputs.erase(name);
    }

    //////////////////////////////////////////////////////////////////////////
    void Debug::LogAllOutputs(bool enable)
    {
        g_LogAllOutputs = enable;
    }

    //////////////////////////////////////////////////////////////////////////
    void Debug::LogGrad(const string& name, bool enable)
    {
        if (enable)
            g_LogGrads.insert(name);
        else
            g_LogGrads.erase(name);
    }

    //////////////////////////////////////////////////////////////////////////
    void Debug::LogAllGrads(bool enable)
    {
        g_LogAllGrads = enable;
    }

    //////////////////////////////////////////////////////////////////////////
    void Debug::Log(const string& name, bool enable)
    {
        LogOutput(name, enable);
        LogGrad(name, enable);
    }

    //////////////////////////////////////////////////////////////////////////
    bool Debug::ShouldLogOutput(const string& name)
    {
        return g_LogAllOutputs || g_LogOutputs.find(name) != g_LogOutputs.end();
    }

    //////////////////////////////////////////////////////////////////////////
    bool Debug::ShouldLogGrad(const string& name)
    {
        return g_LogAllGrads || g_LogGrads.find(name) != g_LogGrads.end();
    }
}