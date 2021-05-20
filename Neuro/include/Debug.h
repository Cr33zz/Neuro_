#pragma once

#include <unordered_set>

#include "Types.h"

#pragma warning(push)
#pragma warning(disable:4251)

//#ifdef LOG_OUTPUTS
//#define _dump(x) dump(x)
//#define _dumpn(x, name) dump(x, name)
//#else
//#define _dump(x) x
//#define _dumpn(x, name) x
//#endif

namespace Neuro
{
    using namespace std;

    class NEURO_DLL_EXPORT Debug
    {
    public:
        static void Step();
        static int GetStep();

        static void LogOutput(const string& name, bool enable = true);
        static void LogAllOutputs(bool enable = true);
        static void LogGrad(const string& name, bool enable = true);
        static void LogAllGrads(bool enable = true);
        static void Log(const string& name, bool enable = true);

        static bool ShouldLogOutput(const string& name);
        static bool ShouldLogGrad(const string& name);

    private:
        static int g_Step;
        static vector<string> g_LogOutputs;
        static vector<string> g_LogGrads;
        static bool g_LogAllOutputs;
        static bool g_LogAllGrads;
    };
}

#pragma warning(pop)