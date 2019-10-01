#include "ComputationalGraph/NameScope.h"

namespace Neuro
{
    list<string> NameScope::s_Scopes;

    //////////////////////////////////////////////////////////////////////////
    NameScope::NameScope(const string& scopeName)
    {
        s_Scopes.push_back(scopeName);
    }

    //////////////////////////////////////////////////////////////////////////
    NameScope::~NameScope()
    {
        s_Scopes.pop_back();
    }

    //////////////////////////////////////////////////////////////////////////
    string NameScope::Name()
    {
        string name;
        for (auto& scope : s_Scopes)
        {
            name.append(scope);
            name.append("/");
        }
        return name;
    }
}