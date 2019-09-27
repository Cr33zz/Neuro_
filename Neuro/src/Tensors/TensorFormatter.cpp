#include <cstdlib>

#include "Tensors/TensorFormatter.h"
#include "Tensors/Tensor.h"
#include "Tools.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    string TensorFormatter::ToString(const Tensor& t)
    {
        return ToStringRecursive(t, {}, " ", 75, Dragon4FloatFormatter(t.GetValues()));
    }

    //////////////////////////////////////////////////////////////////////////
    string TensorFormatter::ToStringRecursive(const Tensor& t, const vector<int>& index, const string& hanging_indent, int curr_width, const Dragon4FloatFormatter& formatter)
    {
        string separator = " ";

        int axis = (int)index.size();
        int axes_left = t.GetShape().NDim - axis;

        if (axes_left == 0)
            return formatter.FormatValue(t.GetFlat(t.GetShape().GetIndexKeras(index)));

        // when recursing, add a space to align with the [ added, and reduce the
        // length of the line by 1
        string next_hanging_indent = hanging_indent + ' ';
        int next_width = curr_width - 1;

        int a_len = t.GetShape().Dimensions[t.GetShape().NDim - 1 - axis];
        int leading_items = 0;
        int trailing_items = a_len;

        // stringify the array with the hanging indent on the first line too
        string s = "";

        // last axis (rows) - wrap elements if they would not fit on one line
        if (axes_left == 1)
        {
            // the length up until the beginning of the separator / bracket
            int elem_width = curr_width - 1;

            string line = hanging_indent;
            string word = "";

            for (int i = 0; i < leading_items; ++i)
            {
                vector<int> tempIndex = index;
                tempIndex.push_back(i);
                word = ToStringRecursive(t, tempIndex, next_hanging_indent, next_width, formatter);
                auto exLine = ExtendLine(s, line, word, elem_width, hanging_indent);
                s = exLine.first;
                line = exLine.second;
                line += separator;
            }

            for (int i = trailing_items; i > 1; --i)
            {
                vector<int> tempIndex = index;
                tempIndex.push_back(-i);
                word = ToStringRecursive(t, tempIndex, next_hanging_indent, next_width, formatter);
                auto exLine = ExtendLine(s, line, word, elem_width, hanging_indent);
                s = exLine.first;
                line = exLine.second;
                line += separator;
            }

            vector<int> tempIndex = index;
            tempIndex.push_back(-1);
            word = ToStringRecursive(t, tempIndex, next_hanging_indent, next_width, formatter);
            auto exLine = ExtendLine(s, line, word, elem_width, hanging_indent);
            s = exLine.first;
            line = exLine.second;

            s += line;
        }
        // other axes - insert newlines between rows
        else
        {
            string line_sep = separator;
            for (int i = 0; i < axes_left - 1; ++i)
                line_sep += '\n';
            string nested = "";

            for (int i = 0; i < leading_items; ++i)
            {
                vector<int> tempIndex = index;
                tempIndex.push_back(i);
                nested = ToStringRecursive(t, tempIndex, next_hanging_indent, next_width, formatter);
                s += hanging_indent + nested + line_sep;
            }

            for (int i = trailing_items; i > 1; --i)
            {
                vector<int> tempIndex = index;
                tempIndex.push_back(-i);
                nested = ToStringRecursive(t, tempIndex, next_hanging_indent, next_width, formatter);
                s += hanging_indent + nested + line_sep;
            }

            vector<int> tempIndex = index;
            tempIndex.push_back(-1);
            nested = ToStringRecursive(t, tempIndex, next_hanging_indent, next_width, formatter);

            s += hanging_indent + nested;
        }

        // remove the hanging indent, and wrap in []
        s = '[' + s.substr(hanging_indent.length()) + ']';

        return s;
    }

    //////////////////////////////////////////////////////////////////////////
    pair<string, string> TensorFormatter::ExtendLine(string& str, string& line, const string& word, int line_width, const string& next_line_prefix)
    {
        bool needs_wrap = (line.length() + word.length()) > line_width;
        if (line.length() <= next_line_prefix.length())
            needs_wrap = false;

        if (needs_wrap)
        {
            str += TrimEnd(line) + "\n";
            line = next_line_prefix;
        }

        line += word;
        return make_pair(str, line);
    }

    //////////////////////////////////////////////////////////////////////////
    TensorFormatter::Dragon4FloatFormatter::Dragon4FloatFormatter(const vector<float>& a)
    {
        for (uint32_t i = 0; i < a.size(); ++i)
        {
            auto s = Split(Float2Str(a[i]), ".");
            m_PadLeft = max(m_PadLeft, s[0].length());
            m_PadRight = max(m_PadRight, s.size() > 1 ? s[1].length() : 0);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    string TensorFormatter::Dragon4FloatFormatter::FormatValue(float v) const
    {
        auto s = Split(Float2Str(v), ".");
        stringstream stream;
        stream << right << setw(m_PadLeft) << s[0] << "." << left << setw(m_PadRight) << (s.size() > 1 ? s[1] : "");
        return stream.str();
    }

    //////////////////////////////////////////////////////////////////////////
    string TensorFormatter::Dragon4FloatFormatter::Float2Str(float f, int precision) const
    {
        stringstream stream;
        stream << fixed << setprecision(5) << f;
        return stream.str();
    }
}