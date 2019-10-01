#pragma once

#include <list>
#include <string>

namespace Neuro
{
    using namespace std;

    class NameScope
    {
    public:
        NameScope(const string& scopeName);
        ~NameScope();

        static string Name();

    private:
        static list<string> s_Scopes;
    };
}
