#pragma once

#include <string>

using namespace std;

class Args
{
public:
    Args(int argc, char *argv[])
    {
        string lastArgName = "";

        for (int i = 1; i < argc; ++i)
        {
            string arg = argv[i];

            if (arg[0] == '-')
            {
                if (lastArgName != "")
                    m_Args[lastArgName] = "__present";

                lastArgName = arg.substr(1);
            }
            else
            {
                if (lastArgName != "")
                {
                    m_Args[lastArgName] = arg;
                    lastArgName = "";
                }
                else
                    NEURO_ASSERT(false, "Invalid argument '" + arg + "'");
            }
        }
    }

    string GetArg(const string& name) const
    {
        auto it = m_Args.find(name);

        if (it == m_Args.end())
            return "";

        return it->second;
    }

    float GetArgFloat(const string& name) const
    {
        return stof(GetArg(name));
    }

    int GetArgInt(const string& name) const
    {
        return stoi(GetArg(name));
    }

    bool HasArg(const string& name) const
    {
        return m_Args.find(name) != m_Args.end();
    }

private:
    map<string, string> m_Args;
};