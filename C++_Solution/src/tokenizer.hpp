#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <vector>
#include <string>

namespace Tokenizer {
    // one hot word embeddings
    std::vector<std::vector<int>> oneHot(std::vector<std::string> input, std::vector<std::string> vocab);

    // unique words in a set of words
    std::vector<std::string> unique(std::vector<std::string> input);
}

#endif // TOKENIZER_H_
