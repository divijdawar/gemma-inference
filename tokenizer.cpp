#include <string> 
#include <vector> 
#include <sentencepiece_processor.h> 
#include <stdexcept>

class Tokenizer {
private: 
    sentencepiece::SentencePieceProcessor sp; 
    int bos_id, eos_id, pad_id, unk_id; 
    int max_len = 8192; 
    std::string model_path; 

public: 
    Tokenizer(const std::string& model_path) : model_path(model_path) { 
        auto status = sp.Load(model_path); 
        if (!status.ok()) { 
            throw std::runtime_error("Failed to load tokenizer model"); 
        }
        bos_id = sp.PieceToId("<bos>"); 
        eos_id = sp.PieceToId("<eos>"); 
        pad_id = sp.PieceToId("<pad>"); 
        unk_id = sp.unk_id(); 
    }

    std::vector<std::vector<int>> encode(
        const std::vector<std::string>& id, 
        bool special_tokens = true, 
        bool padding = false, 
        int max_len_override = -1
    )  {
        const int effective_max_len = (max_len_override == -1) ? max_len : max_len_override;
        
        std::vector<std::vector<int>> encoded_input; 

        for (const auto& t : id) { 
            std::vector<int> ids; 
            sp.Encode(t, &ids); 
            
            if (special_tokens) { 
                ids.insert(ids.begin(), bos_id); 
                ids.push_back(eos_id); 
            }
            
            if (padding && static_cast<int>(ids.size()) < effective_max_len) { 
                int padding_needed = effective_max_len - static_cast<int>(ids.size()); 
                ids.insert(ids.end(), padding_needed, pad_id);
            }
            
            encoded_input.push_back(ids); 
        }
        return encoded_input; 
    }
    
};