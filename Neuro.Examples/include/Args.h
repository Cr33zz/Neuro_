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

        if (lastArgName != "")
            m_Args[lastArgName] = "__present";
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

    vector<string> GetArgArray(const string& name) const
    {
        return Tokenize(GetArg(name));
    }

    vector<float> GetArgArrayFloat(const string& name) const
    {
        vector<float> args;
        for (auto& token : Tokenize(GetArg(name)))
            args.push_back(stof(token));
        return args;
    }

    template <typename INT_TYPE = int>
    vector<INT_TYPE> GetArgArrayInt(const string& name) const
    {
        vector<INT_TYPE> args;
        for (auto& token : Tokenize(GetArg(name)))
            args.push_back(stoi(token));
        return args;
    }

private:
    vector<string> Tokenize(const string& str) const
    {
        vector<string> tokens;
        vector<char> strData(begin(str), end(str));
        char* context = nullptr;

        char* pch = strtok_s(strData.data(), ",", &context);
        while (pch)
        {
            tokens.push_back(pch);
            pch = strtok_s(nullptr, ",", &context);
        }

        return tokens;
    }

    map<string, string> m_Args;
};