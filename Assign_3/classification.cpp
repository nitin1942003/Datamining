#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <numeric>  // Add this for inner_product
using namespace std;

class MatrixCls {
private:
    vector<vector<string>> Matrix;
public:
    MatrixCls(string Data_File) {
        ifstream Data(Data_File);
        string line, item;
        vector<string> Row;
        while(getline(Data, line)) {
            istringstream iss(line);
            while(iss >> item && item.length()) Row.push_back(item);
            if(line.length()) {
                Matrix.push_back(Row);
                Row.clear();
            }
        }
        Data.close();
    }

    MatrixCls(vector<vector<string>> A_Matrix) : Matrix(A_Matrix) {}
    MatrixCls() {}
    ~MatrixCls() {}

    string Element(int i, int j) const { return Matrix[i][j]; }
    int SizeX() const { return Matrix[0].size(); }
    int SizeY() const { return Matrix.size(); }

    vector<string> GetAttributes() const {
        return vector<string>(Matrix[0].begin(), Matrix[0].end() - 1);
    }

    map<string, vector<string>> GetAttributesValues() const {
        map<string, vector<string>> Attributes_Values;
        for(int j = 0; j < SizeX(); j++) {
            vector<string> values;
            for(int i = 1; i < SizeY(); i++) values.push_back(Matrix[i][j]);
            sort(values.begin(), values.end());
            values.erase(unique(values.begin(), values.end()), values.end());
            Attributes_Values[Matrix[0][j]] = values;
        }
        return Attributes_Values;
    }

    vector<string> GetAttributeValues(string The_Attribute) const {
        return GetAttributesValues()[The_Attribute];
    }

    vector<string> GetScoreRange() const {
        return GetAttributesValues()[Matrix[0].back()];
    }

    int AttributeIndex(string The_Attribute) const {
        auto it = find(Matrix[0].begin(), Matrix[0].end(), The_Attribute);
        return it - Matrix[0].begin();
    }

    map<string, vector<string>> GetAttributeValuesScores(string The_Attribute) const {
        int idx = AttributeIndex(The_Attribute);
        map<string, vector<string>> result;
        auto values = GetAttributeValues(The_Attribute);
        for(const auto& val : values) {
            vector<string> scores;
            for(int i = 1; i < SizeY(); i++)
                if(Matrix[i][idx] == val)
                    scores.push_back(Matrix[i].back());
            result[val] = scores;
        }
        return result;
    }

    vector<string> GetScores() const {
        vector<string> scores;
        for(int i = 1; i < SizeY(); i++)
            scores.push_back(Matrix[i].back());
        return scores;
    }

    MatrixCls operator()(const MatrixCls& A_Matrix, string The_Attribute, string The_Value) {
        int idx = A_Matrix.AttributeIndex(The_Attribute);
        vector<vector<string>> newMatrix;
        vector<string> header;
        
        for(int j = 0; j < A_Matrix.SizeX(); j++)
            if(j != idx) header.push_back(A_Matrix.Element(0,j));
        if(!header.empty()) newMatrix.push_back(header);

        for(int i = 1; i < A_Matrix.SizeY(); i++) {
            if(A_Matrix.Element(i,idx) == The_Value) {
                vector<string> row;
                for(int j = 0; j < A_Matrix.SizeX(); j++)
                    if(j != idx) row.push_back(A_Matrix.Element(i,j));
                if(!row.empty()) newMatrix.push_back(row);
            }
        }
        Matrix = newMatrix;
        return *this;
    }

    void Display() const {
        for(const auto& row : Matrix) {
            for(const auto& elem : row) cout << " " << elem;
            cout << endl;
        }
    }
};

vector<string> GetUniqueScores(vector<string> Scores) {
    sort(Scores.begin(), Scores.end());
    Scores.erase(unique(Scores.begin(), Scores.end()), Scores.end());
    return Scores;
}

string GetFrequentScore(const vector<string>& Scores) {
    map<string,int> freq;
    for(const auto& score : Scores) freq[score]++;
    return max_element(freq.begin(), freq.end(),
        [](const pair<string,int>& a, const pair<string,int>& b) {
            return a.second < b.second;
        })->first;
}

double ComputeEntropy(const vector<string>& Scores) {
    if(Scores.empty()) return 0.0;
    
    map<string,int> freq;
    for(const auto& score : Scores) freq[score]++;
    
    double entropy = 0.0;
    double size = Scores.size();
    for(const auto& pair : freq) {
        double p = pair.second / size;
        entropy -= p * log2(p);
    }
    return entropy;
}

double ComputeAttributeEntropyGain(const MatrixCls& Remain_Matrix, string The_Attribute) {
    auto scores = Remain_Matrix.GetScores();
    double originalEntropy = ComputeEntropy(scores);
    
    auto valuesScores = Remain_Matrix.GetAttributeValuesScores(The_Attribute);
    double afterEntropy = 0.0;
    for(const auto& pair : valuesScores) {
        afterEntropy += ComputeEntropy(pair.second) * pair.second.size() / scores.size();
    }
    return originalEntropy - afterEntropy;
}

class Tree {
public:
    string Node, Branch;
    vector<Tree*> Child;
    Tree() : Node(""), Branch("") {}
    
    string Temp_TestTree(Tree* tree, const vector<string>& Attributes, 
                        const vector<string>& Values, const vector<string>& Score_Range) {
        auto it = find(Score_Range.begin(), Score_Range.end(), tree->Node);
        if(it != Score_Range.end()) return tree->Node;

        auto attrIt = find(Attributes.begin(), Attributes.end(), tree->Node);
        if(attrIt != Attributes.end()) {
            int idx = attrIt - Attributes.begin();
            for(auto child : tree->Child) {
                if(child->Branch == Values[idx]) {
                    if(find(Score_Range.begin(), Score_Range.end(), child->Node) != Score_Range.end())
                        return child->Node;
                        
                    vector<string> newAttrs, newVals;
                    for(size_t i = 0; i < Attributes.size(); i++)
                        if(i != idx) {
                            newAttrs.push_back(Attributes[i]);
                            newVals.push_back(Values[i]);
                        }
                    return Temp_TestTree(child, newAttrs, newVals, Score_Range);
                }
            }
            return GetFrequentScore(Score_Range);
        }
        return Score_Range.empty() ? "unknown" : Score_Range[0];
    }

    Tree* BuildTree(Tree* tree, const MatrixCls& Remain_Matrix) {
        if(!tree) tree = new Tree();
        
        auto scores = Remain_Matrix.GetScores();
        auto uniqueScores = GetUniqueScores(scores);
        
        if(uniqueScores.size() == 1) {
            tree->Node = uniqueScores[0];
            return tree;
        }
        
        if(Remain_Matrix.SizeX() == 1) {
            tree->Node = GetFrequentScore(scores);
            return tree;
        }
        
        auto attributes = Remain_Matrix.GetAttributes();
        string maxAttr;
        double maxGain = 0;
        
        for(const auto& attr : attributes) {
            double gain = ComputeAttributeEntropyGain(Remain_Matrix, attr);
            if(gain > maxGain) {
                maxGain = gain;
                maxAttr = attr;
            }
        }
        
        tree->Node = maxAttr;
        auto values = Remain_Matrix.GetAttributeValues(maxAttr);
        
        for(const auto& val : values) {
            MatrixCls newMatrix;
            newMatrix = newMatrix(Remain_Matrix, maxAttr, val);
            Tree* child = new Tree();
            child->Branch = val;
            
            if(newMatrix.SizeX() == 1)
                child->Node = GetFrequentScore(newMatrix.GetScores());
            else
                BuildTree(child, newMatrix);
                
            tree->Child.push_back(child);
        }
        return tree;
    }

    void PrintTree(Tree* tree, int depth = -1) {
        string indent(max(0, depth), '\t');
        if(!tree->Branch.empty()) {
            cout << indent << tree->Branch << endl;
            indent += "\t";
        }
        if(depth == -1 && !tree->Branch.empty()) cout << "\t";
        cout << indent << tree->Node << endl;
        for(auto child : tree->Child) PrintTree(child, depth + 1);
    }

    vector<string> TestTree(Tree* tree, const MatrixCls& The_Matrix) {
        vector<string> results;
        auto attrs = The_Matrix.GetAttributes();
        auto range = The_Matrix.GetScoreRange();
        
        for(int i = 1; i < The_Matrix.SizeY(); i++) {
            vector<string> values;
            for(int j = 0; j < attrs.size(); j++)
                values.push_back(The_Matrix.Element(i,j));
            results.push_back(Temp_TestTree(tree, attrs, values, range));
        }
        return results;
    }
};

int main() {
    MatrixCls TrainMatrix("Train.dat");
    Tree* root = new Tree();
    root = root->BuildTree(root, TrainMatrix);
    
    cout << "Decision Tree Structure:" << endl;
    root->PrintTree(root);

    MatrixCls TestMatrix("Test.dat");
    auto testScores = root->TestTree(root, TestMatrix);
    auto origScores = TestMatrix.GetScores();

    cout << "\nOriginal_Scores (from Test.dat):\n";
    for(const auto& score : origScores) cout << score << "  ";
    
    cout << "\n\nPredicted_Scores (from Test.dat):\n";
    for(const auto& score : testScores) cout << score << "  ";
    
    // Calculate accuracy using a simple loop
    int correct = 0;
    for(size_t i = 0; i < testScores.size(); i++) {
        if(testScores[i] == origScores[i]) {
            correct++;
        }
    }
    
    double accuracy = 100.0 * correct / testScores.size();
    cout << "\n\nAccuracy on test data: " << accuracy << "%" << endl;
    
    delete root;
}