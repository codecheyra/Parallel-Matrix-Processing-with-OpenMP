module load compiler/gcc/9.1.0
module load compiler/gcc/9.1/mpich/3.3.1
g++ -O3 -fopenmp -std=c++17 check.cpp template.cpp -o check && ./check