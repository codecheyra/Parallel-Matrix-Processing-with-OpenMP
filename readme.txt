In my code i have used :

```
random_device rd;
mt19937 gen(rd());
shuffle(block_indices.begin(), block_indices.end(), gen);
block_indices.resize(b);
uniform_int_distribution<> dis(0, 256);
```
because random_device is a uniformly-distributed random number generator that is used to seed a pseudo-random generator,
mt19937 is a fast pseudo-random number generator. This generates fully random 32-bit or 64-bit integers. 

I have tried optimising my code at blocks multiplication with the permutations and combinations of ijk loops done as in our assignment0
but it was not giving me the correct output so i have used the simple ijk loop for the multiplication of blocks.

secondly, i have tried writing as SIMD architecture but my row statistics were not matching 
correctly so i havent done with this approach.

Nmae: Kethavath Ajay Kumar
Entry No: 2021CS11211