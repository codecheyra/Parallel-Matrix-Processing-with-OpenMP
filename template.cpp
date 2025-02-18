#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <omp.h>
#include "check.h"
// #include <chrono>
using namespace std;
// int NUM_THREADS = 40;
map<pair<int, int>, vector<vector<int>>> generate_matrix(int n, int m, int b) {
    map<pair<int, int>, vector<vector<int>>> matrix_map;
    int total_blocks = (n / m) * (n / m);

    if (total_blocks < b) {
        cout << "Number of blocks is greater than the total number of blocks possible" << endl;
        return matrix_map;
    }

    vector<pair<int, int>> block_indices;
    for (int i = 0; i < n / m; ++i) {
        for (int j = 0; j < n / m; ++j) {
            block_indices.emplace_back(i, j);
        }
    }

    random_device rd;
    mt19937 gen(rd());
    shuffle(block_indices.begin(), block_indices.end(), gen);
    block_indices.resize(b);

    uniform_int_distribution<> dis(0, 256);

    // omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (const auto& block_index : block_indices) {
                #pragma omp task shared(matrix_map) if(black_box())
                {
                    vector<vector<int>> block(m, vector<int>(m));
                    for (int i = 0; i < m; ++i) {
                        for (int j = 0; j < m; ++j) {
                            block[i][j] = dis(gen);
                        }
                    }
                    #pragma omp critical
                    {
                        matrix_map[block_index] = block;
                    }
                }
            }
        }
    }

    return matrix_map;
}

void preprocess_matrix(map<pair<int, int>, vector<vector<int>>>& blocks, int m) {
    for (auto it = blocks.begin(); it != blocks.end(); ) {
        auto& block = it->second;
        bool isBlockNonZero = false;

        for (auto& row : block) {
            for (auto& val : row) {
                if (val % 5 == 0) {
                    val = 0;
                }
                if (val != 0) {
                    isBlockNonZero = true;
                }
            }
        }

        if (!isBlockNonZero) {
            it = blocks.erase(it);
        } else {
            ++it;
        }
    }
}

vector<vector<int>> multiply_blocks1(const vector<vector<int>>& block1, const vector<vector<int>>& block2, int m) {
    vector<vector<int>> result(m, vector<int>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                result[i][j] += block1[i][k] * block2[k][j];
            }
        }
    }
    return result;
}

vector<float> matmul(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    // cout << "Matrix multiplication started" << endl;
    // std::chrono::time_point<chrono::system_clock> start, end;
    // start = chrono::system_clock::now();
    preprocess_matrix(blocks, m);

    if (blocks.empty()) {
        return vector<float>(n, 0.0f);
    }

    vector<float> row_statistics(n, 0.0f);
    vector<int> P(n, 0);
    vector<int> B(n, 0);

    map<pair<int, int>, vector<vector<int>>> result_blocks = blocks;

    int power = 1;
    while (power < k) {
        map<pair<int, int>, vector<vector<int>>> temp_result_blocks;
        bool any_multiplication = false;

        // omp_set_num_threads(NUM_THREADS);
        #pragma omp parallel 
        {
            #pragma omp single
            {
                for (const auto& [block_pos1, block1] : result_blocks) {
                    for (const auto& [block_pos2, block2] : blocks) {
                        if (block_pos1.second == block_pos2.first) {
                            #pragma omp task shared(temp_result_blocks, P) if(black_box())
                            {
                                auto product = multiply_blocks1(block1, block2, m);

                                #pragma omp critical
                                {
                                    auto& result_block = temp_result_blocks[{block_pos1.first, block_pos2.second}];
                                    if (result_block.empty()) {
                                        result_block = product;
                                    } else {
                                        for (int i = 0; i < m; ++i) {
                                            for (int j = 0; j < m; ++j) {
                                                result_block[i][j] += product[i][j];
                                            }
                                        }
                                    }
                                }

                                any_multiplication = true;
                                if (power == 1 && k == 2) {
                                    for (int i = 0; i < m; i++) {
                                        for (int j = 0; j < m; j++) {
                                            for (int f = 0; f < m; f++) {
                                                int val = block1[i][f] * block2[f][j];
                                                if (val != 0) {
                                                    #pragma omp atomic
                                                    P[block_pos1.first * m + i] += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #pragma omp taskwait

        if (!any_multiplication) break;

        result_blocks = temp_result_blocks;
        power++;
    }
    blocks = result_blocks;
    for (const auto& [block_pos, block] : blocks) {
        for (int i = 0; i < m; i++) {
            B[block_pos.first * m + i] += m;
        }
    }

    if (k == 2) {
        for (int i = 0; i < n; i++) {
            if (B[i] != 0) {
                row_statistics[i] = static_cast<float>(P[i]) / B[i];
            }
        }
    }
    // end = chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // cout << "Matrix multiplication ended" << endl;
    // cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    
    return (k == 2) ? row_statistics : vector<float>();
    
}