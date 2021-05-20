#pragma once

#include <list>
#include <string>

#include "Types.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
    using namespace std;

    class NEURO_DLL_EXPORT NameScope
    {
    public:
        NameScope(const string& scopeName);
        ~NameScope();

        static string Name();

    private:
        static list<string> s_Scopes;
    };
}

#pragma warning(pop)