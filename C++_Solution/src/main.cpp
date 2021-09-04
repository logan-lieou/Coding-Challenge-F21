#include "tokenizer.hpp"
#include "transformer.hpp"

#include <iostream>
#include <vector>
#include <string>

std::vector<std::string> get_list()
{
    std::vector<std::string> a = {"list", "of", "strings"};
    return a;
}

template<typename T>
void outVector(std::vector<T> vec)
{
    for(auto i = vec.cbegin(); i < vec.cend(); i++)
    {
        std::cout << *i << std::endl;
    }
}

int main()
{
    std::vector<std::string> a = get_list();
    std::vector<std::vector<int>> b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::cout << b[0].size() << std::endl;
}
