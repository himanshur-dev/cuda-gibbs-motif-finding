#include <memory>
#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <set>
#include <limits>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include "main.h"



// ----------- User Parameter Logic ----------
struct UserArgs {
    char *inputFileName;
    int numIters;
    int kmerSize;
    int seed = 21; // arbitrairly chosen
    bool useParallel = false;
    bool verboseOutput = false;
    bool printFinalKmers = false;
};
std::shared_ptr<UserArgs> parse_arguments(int argc, char* argv[]) {
    static struct option longOptions[] = {
        {"input-file-name", required_argument, 0, 'i'},
        {"num-iters", required_argument, 0, 'n'},
        {"kmer-size", required_argument, 0, 'k'},
        {"seed", required_argument, 0, 's'},
        {"use-parallel", no_argument, 0, 'p'},
        {"verbose-output", no_argument, 0, 'v'},
        {"print-final-kmers", no_argument, 0, 'f'},
        {0, 0, 0, 0}
    };

    std::shared_ptr<UserArgs> userArgs = std::make_shared<UserArgs>();
    int optionIndex = 0;
    int option;
    while (true) {
        option = getopt_long(argc, argv, "i:n:k:s:pvf", longOptions, &optionIndex);
        if (option == -1) break;
        switch (option) {
            case 'i':
                userArgs->inputFileName = optarg;
                break;
            case 'n':
                userArgs->numIters = atoi(optarg);
                break;
            case 'k':
                userArgs->kmerSize = atoi(optarg);
                break;
            case 's':
                userArgs->seed = atoi(optarg);
                break;
            case 'p':
                userArgs->useParallel = true;
                break;
            case 'v':
                userArgs->verboseOutput = true;
                break;
            case 'f':
                userArgs->printFinalKmers = true;
                break;
            default:
                break;
        }
    }

    return userArgs;
}



// ----------- Data Input Logic ----------
// first line has (num sequences, seq len), following 
//   lines have 
// this fills a 1D array for CUDA compatibility
struct DataWrapper {
    int numSequences;
    int seqLen;
    char *data;
};
std::shared_ptr<DataWrapper> readInputFile(char *inputFileName) {
    std::ifstream file(inputFileName);
    assert(file.is_open());

    int numSequences, seqLen;
    file >> numSequences;
    file >> seqLen;

    std::shared_ptr<DataWrapper> dataWrapper = std::make_shared<DataWrapper>();
    dataWrapper->numSequences = numSequences;
    dataWrapper->seqLen = seqLen;
    // printf("num seqs and seq len: %d, %d \n", numSequences, seqLen);
    dataWrapper->data = new char[numSequences * seqLen];

    std::string sequence;
    int idx = 0;
    while (file >> sequence) {
        assert(sequence.length() == static_cast<size_t>(seqLen));
        for (char nucleotide: sequence) {
            dataWrapper->data[idx] = nucleotide;
            idx++;
        }
    }

    assert(idx == numSequences * seqLen);
    return dataWrapper;
}




// ----------- Sequential Code ----------
// randomly selects a set of indices for the starting kmers
// the selected kmers are in the form [start(1), start(2), ..., start(n)]
void sequential_selectRandomKmers(std::shared_ptr<DataWrapper> dw,
                                  int kmerSize,
                                  int *kmerLocations) {
    for (int i = 0; i < dw->numSequences; i++) {
        int randStart = rand() % (dw->seqLen - kmerSize + 1);
        assert(randStart + kmerSize <= dw->seqLen);
        kmerLocations[i] = randStart;
    }
}
void printKmers(std::shared_ptr<DataWrapper> dw,
                int kmerSize,
                int *kmerLocations) {
    printf("Selected kmers: \n");
    for (int currSeqIdx = 0; currSeqIdx < dw->numSequences; currSeqIdx++) {
        printf("%d: ", currSeqIdx + 1);
        for (int currNucleotideIdx = 0; currNucleotideIdx < dw->seqLen; currNucleotideIdx++) {
            char currNucleotide = dw->data[(currSeqIdx * dw->seqLen) + currNucleotideIdx];
            if (currNucleotideIdx >= kmerLocations[currSeqIdx] && 
                currNucleotideIdx < kmerLocations[currSeqIdx] + kmerSize) {
                printf("%c", currNucleotide);
            } else {
                currNucleotide += 32; // to convert to lowercase
                printf("%c", currNucleotide);
            }
        }
        printf("\n");
    }
}

// return a random int in range [0, max)
int sequential_chooseStringToIgnore(int max) {
    assert(max >= 0);
    int val = rand() % max;
    return val;
}

// constructs a profile matrix w/ Laplace's Rule of Succession
// profile matrix is 1D array with structure [numAs1, numCs1, numGs1, numTs1, 
// numAs2, numCs2, numGs2, numTs2, ..., numAsK, numCsK, numGsK, numTsK]
// note that this is not normalized to a probability distribution for 
// numerical stability, so that will have to be handled downstream
void sequential_constructProfileMatrix(std::shared_ptr<UserArgs> userArgs,
                                       std::shared_ptr<DataWrapper> dw,
                                       int *profile,
                                       int *kmerLocations,
                                       int seqToIgnore) {
    // apply Laplace's Rule of Succession
    for (int i = 0; i < userArgs->kmerSize * NUM_SYMBOLS; i++) {
        profile[i] = 1;
    }
                        
    for (int currSeqIdx = 0; currSeqIdx < dw->numSequences; currSeqIdx++) {
        if (currSeqIdx == seqToIgnore) continue;
        
        int kmerLoc = kmerLocations[currSeqIdx];
        for (int currNucleotideIdx = 0; currNucleotideIdx < userArgs->kmerSize; currNucleotideIdx++) {
            int totalIndex = (currSeqIdx * dw->seqLen) + kmerLocations[currSeqIdx] + currNucleotideIdx;
            assert(totalIndex < dw->numSequences * dw->seqLen);
            char currNucleotide = dw->data[totalIndex];
            switch(currNucleotide) {
                case 'A':
                    profile[(currNucleotideIdx * NUM_SYMBOLS)]++;
                    break;
                case 'C':
                    profile[(currNucleotideIdx * NUM_SYMBOLS)+1]++;
                    break;
                case 'G':
                    profile[(currNucleotideIdx * NUM_SYMBOLS)+2]++;
                    break;
                case 'T':
                    profile[(currNucleotideIdx * NUM_SYMBOLS)+3]++;
                    break;
                default:
                    assert(false);
                    break;
            }
        }
    } 
}
void printProfileMatrix(int *profile, int profileMatrixSize) {
    printf("Profile matrix: \n");
    for (int symbolIdx = 0; symbolIdx < NUM_SYMBOLS; symbolIdx++) {
        char symbol;
        switch(symbolIdx) {
                case 0:
                    symbol = 'A';
                    break;
                case 1:
                    symbol = 'C';
                    break;
                case 2:
                    symbol = 'G';
                    break;
                case 3:
                    symbol = 'T';
                    break;
                default:
                    assert(false);
                    break;
        }
        printf("%c: ", symbol);
        for (int valIdx = symbolIdx; valIdx < profileMatrixSize; valIdx += NUM_SYMBOLS) {
            printf("%d, ", profile[valIdx]);
        }
        printf("\n");
    }
}

void sequential_generateProfileProbabilities(std::shared_ptr<UserArgs> userArgs,
                                             std::shared_ptr<DataWrapper> dw,
                                             int *profile,
                                             int seqToIgnore,
                                             int *profileGeneratedProbabilities,
                                             int numProfileProbabilities) {
    char *seq = &(dw->data[seqToIgnore * dw->seqLen]);
    for (int pos = 0; pos < numProfileProbabilities; pos++) {
        int currProbability = 1;
        for (int subPos = 0; subPos < userArgs->kmerSize; subPos++) {
            int nucleotideIdx = pos + subPos;
            assert(nucleotideIdx < dw->seqLen);
            char currNucleotide = seq[nucleotideIdx];
            switch(currNucleotide) {
                case 'A':
                    currProbability *= profile[(subPos * NUM_SYMBOLS)];
                    break;
                case 'C':
                    currProbability *= profile[(subPos * NUM_SYMBOLS)+1];
                    break;
                case 'G':
                    currProbability *= profile[(subPos * NUM_SYMBOLS)+2];
                    break;
                case 'T':
                    currProbability *= profile[(subPos * NUM_SYMBOLS)+3];
                    break;
                default:
                    assert(false);
                    break;
            }
        }
        profileGeneratedProbabilities[pos] = currProbability;
    }
}
void printProfileProbabilities(int *profileGeneratedProbabilities, int numProfileProbabilities) {
    printf("Profile probabilities for ignored sequence: \n");
    for (int i = 0; i < numProfileProbabilities; i++) {
        printf("%d: %d\n", i+1, profileGeneratedProbabilities[i]);
    }
}

int sequential_pickRandomKmer(int *profileGeneratedProbabilities, int numProfileProbabilities) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(profileGeneratedProbabilities, profileGeneratedProbabilities + numProfileProbabilities);
    int index = dist(gen);
    assert(index >=0 && index < numProfileProbabilities);
    return index;
}

// this uses the Hamming distance as the distace metric
int sequential_calculateKmerScore(std::shared_ptr<UserArgs> userArgs,
                                  std::shared_ptr<DataWrapper> dw,
                                  int *kmerLocations) {
    int score = 0;
    std::vector<int> counts; 
    for (int currPos = 0; currPos < userArgs->kmerSize; currPos++) {
        counts = {0, 0, 0, 0};
        for (int seqIdx = 0; seqIdx < dw->numSequences; seqIdx++) {
            int currNucleotideIdx = (seqIdx * dw->seqLen) + kmerLocations[seqIdx] + currPos;
            char currNucleotide = dw->data[currNucleotideIdx];
            switch(currNucleotide) {
                case 'A':
                    counts[0]++;
                    break;
                case 'C':
                    counts[1]++;
                    break;
                case 'G':
                    counts[2]++;
                    break;
                case 'T':
                    counts[3]++;
                    break;
                default:
                    assert(false);
                    break;
            }
        }
        score += dw->numSequences - *std::max_element(counts.begin(), counts.end()); // this
        // is the total - max count, which gives us the # of nucleotides not in the consensus
        // sequence 
    }
    return score; 
}


// ----------- General Utility Functions -----------
void copyKmers(int *source, int *dest, int numSequences) {
    for (int i = 0; i < numSequences; i++) {
        dest[i] = source[i];
    }
}



// ----------- Main Function ----------
int main(int argc, char* argv[]) {
    printf("\n\n");
    std::shared_ptr<UserArgs> userArgs = parse_arguments(argc, argv);
    srand(userArgs->seed); // use this to have RNG consistency
    std::shared_ptr<DataWrapper> dataWrapper = readInputFile(userArgs->inputFileName);

    // print a bunch of user-inputted stuff
    if (userArgs->verboseOutput) {
        printf("Input file: %s\n", userArgs->inputFileName);
        printf("Number of iterations: %d\n", userArgs->numIters);
        printf("K-mer size: %d\n", userArgs->kmerSize);
        printf("Running in parallel mode?: %s\n", userArgs->useParallel ? "true" : "false");
        printf("Number of sequences: %d\n", dataWrapper->numSequences);
        printf("Length of each sequence: %d\n", dataWrapper->seqLen);
        printf("\n\n");
    }

    // timing stuff
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
    startTime = std::chrono::steady_clock::now();

    // randomly select k-mers from each string in sequences
    // we don't parallelize this since it's a one-time operation
    int kmerLocationsSize = dataWrapper->numSequences;
    int *kmerLocations = new int[kmerLocationsSize];
    int *bestKmerLocations = new int[kmerLocationsSize];
    sequential_selectRandomKmers(dataWrapper, userArgs->kmerSize, kmerLocations);
    copyKmers(kmerLocations, bestKmerLocations, dataWrapper->numSequences);
    int bestKmerScore = sequential_calculateKmerScore(userArgs, dataWrapper, kmerLocations);
    if (userArgs->verboseOutput) {
        printKmers(dataWrapper, userArgs->kmerSize, kmerLocations);
        printf("\n");
    }

    // profile matrix and probabilities info
    int profileMatrixSize = userArgs->kmerSize * NUM_SYMBOLS;
    int *profileMatrix = new int[profileMatrixSize];
    int numProfileProbabilities = dataWrapper->seqLen - userArgs->kmerSize + 1;
    int *profileGeneratedProbabilities = new int[numProfileProbabilities];

    if (!userArgs->useParallel) {
        // sequential implementation
        if (userArgs->verboseOutput) printf("Starting sequential run\n");

        // in loop:
        for (int i = 0; i < userArgs->numIters; i++) {
            if (userArgs->verboseOutput) printf("-------- Current Iteration: %d --------\n", i+1);
            // 1: randomly select a string to ignore 
            int seqToIgnore = sequential_chooseStringToIgnore(dataWrapper->numSequences);
            if (userArgs->verboseOutput) printf("Ignoring seqence: %d\n", seqToIgnore + 1); 
            // 2: construct a profile matrix for the remaining motifs, using
            //    Laplace's rule of succession
            sequential_constructProfileMatrix(userArgs, dataWrapper, profileMatrix, 
            kmerLocations, seqToIgnore);
            if (userArgs->verboseOutput) printProfileMatrix(profileMatrix, profileMatrixSize);
            // 3: calculate probabilities for each possibe k-mer in 
            //    the deleted sequence
            sequential_generateProfileProbabilities(userArgs, dataWrapper, profileMatrix, seqToIgnore, profileGeneratedProbabilities, numProfileProbabilities);
            if (userArgs->verboseOutput) printProfileProbabilities(profileGeneratedProbabilities, numProfileProbabilities);
            // 4: get random k-mer by using probability distribution over 
            //    k-mer probabilities
            int newKmerStart = sequential_pickRandomKmer(profileGeneratedProbabilities, numProfileProbabilities);
            if (userArgs->verboseOutput) printf("New kmer start position: %d\n", newKmerStart + 1);
            // 5: replace motif for that string
            kmerLocations[seqToIgnore] = newKmerStart;
            if (userArgs->verboseOutput) printKmers(dataWrapper, userArgs->kmerSize, kmerLocations);
            // 6: score the motifs, if they are better then they are the new best motifs
            int currKmerScore = sequential_calculateKmerScore(userArgs, dataWrapper, kmerLocations);
            if (userArgs->verboseOutput) printf("Score for this iteration: %d\n", currKmerScore);
            if (currKmerScore < bestKmerScore) {
                copyKmers(kmerLocations, bestKmerLocations, dataWrapper->numSequences);
                bestKmerScore = currKmerScore;
                if (userArgs->verboseOutput) printf("Found new optimal solution!\n");
            }
            if (userArgs->verboseOutput) printf("--------------------------------------\n\n");
        }
        if (userArgs->verboseOutput) printf("Sequential run finished!\n");
    } else {
        // parallel implementation
        // note that some parts remain sequential, as it doesn't make sense to 
        // parallelize all parts of the algorithm
        if (userArgs->verboseOutput) printf("Starting parallel run\n");

        // allocate GPU memory
        int *d_profileMatrix, *d_kmerLocations, *d_profileGeneratedProbabilities, *d_kmerSubscores;
        char *d_data;
        cudaError_t cErr;
        cErr = cudaMalloc(&d_profileMatrix, profileMatrixSize * sizeof(int));
        assert(cErr == cudaSuccess);
        cErr = cudaMalloc(&d_kmerLocations, kmerLocationsSize * sizeof(int));
        assert(cErr == cudaSuccess);
        cErr = cudaMalloc(&d_profileGeneratedProbabilities, numProfileProbabilities * sizeof(int));
        assert(cErr == cudaSuccess);
        cErr = cudaMalloc(&d_data, dataWrapper->numSequences * dataWrapper->seqLen * sizeof(char));
        assert(cErr == cudaSuccess);
        cErr = cudaMalloc(&d_kmerSubscores, userArgs->kmerSize * sizeof(int));
        assert(cErr == cudaSuccess);
        cErr = cudaMemcpy(d_kmerLocations, kmerLocations, 
            kmerLocationsSize * sizeof(int), cudaMemcpyHostToDevice);
        assert(cErr == cudaSuccess);
        cErr = cudaMemcpy(d_data, dataWrapper->data, 
            dataWrapper->numSequences * dataWrapper->seqLen * sizeof(char), cudaMemcpyHostToDevice);
        assert(cErr == cudaSuccess);


        // in loop:
        for (int i = 0; i < userArgs->numIters; i++) {
            if (userArgs->verboseOutput) printf("-------- Current Iteration: %d --------\n", i+1);
            // 1: randomly select a string to ignore
            int seqToIgnore = sequential_chooseStringToIgnore(dataWrapper->numSequences);
            if (userArgs->verboseOutput) printf("Ignoring seqence: %d\n", seqToIgnore + 1);
            // 2: construct profile matrix in GPU
            parallel_generateProfileMatrix(d_data, d_profileMatrix, 
                d_kmerLocations, seqToIgnore, dataWrapper->numSequences, dataWrapper->seqLen, userArgs->kmerSize);
            if (userArgs->verboseOutput) {
                cErr = cudaMemcpy(profileMatrix, d_profileMatrix, 
                    profileMatrixSize * sizeof(int), cudaMemcpyDeviceToHost);
                assert(cErr == cudaSuccess);
                printProfileMatrix(profileMatrix, profileMatrixSize);
            }
            // 3: calculate kmer probabilities for the deleted sequence
            parallel_generateProfileProbabilities(d_data, d_profileMatrix, d_profileGeneratedProbabilities, numProfileProbabilities,
                seqToIgnore, dataWrapper->seqLen, userArgs->kmerSize);
            cErr = cudaMemcpy(profileGeneratedProbabilities, d_profileGeneratedProbabilities, 
                numProfileProbabilities * sizeof(int), cudaMemcpyDeviceToHost);
            assert(cErr == cudaSuccess);
            if (userArgs->verboseOutput) {
                printProfileProbabilities(profileGeneratedProbabilities, numProfileProbabilities);
            }
            // 4: get random k-mer
            int newKmerStart = sequential_pickRandomKmer(profileGeneratedProbabilities, numProfileProbabilities);
            if (userArgs->verboseOutput) printf("New kmer start position: %d\n", newKmerStart + 1);
            // 5: replace motif
            kmerLocations[seqToIgnore] = newKmerStart;
            cErr = cudaMemcpy(&d_kmerLocations[seqToIgnore], &kmerLocations[seqToIgnore], 
                sizeof(int), cudaMemcpyHostToDevice); // copy over the one changed k-mer for the next iteration 
            assert(cErr == cudaSuccess);
            if (userArgs->verboseOutput) printKmers(dataWrapper, userArgs->kmerSize, kmerLocations);
            // 6: score motifs
            int currKmerScore = parallel_calculateKmerScore(d_data, d_kmerLocations, d_kmerSubscores, dataWrapper->numSequences,
                dataWrapper->seqLen, userArgs->kmerSize);
            if (userArgs->verboseOutput) printf("Score for this iteration: %d\n", currKmerScore);
            if (currKmerScore < bestKmerScore) {
                copyKmers(kmerLocations, bestKmerLocations, dataWrapper->numSequences);
                bestKmerScore = currKmerScore;
                if (userArgs->verboseOutput) printf("Found new optimal solution!\n");
            }
            if (userArgs->verboseOutput) printf("--------------------------------------\n\n");
        }
    }

    // clean up data
    delete[] kmerLocations;
    delete[] bestKmerLocations;
    delete[] profileMatrix;
    delete[] profileGeneratedProbabilities;

    endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(endTime - startTime);

    // Output results
    printf("-------- Final Data --------\n");
    printf("Runtime: %lf\n", time.count());
    printf("Best k-mer score: %d\n", bestKmerScore);
    if (userArgs->printFinalKmers) {
        printKmers(dataWrapper, userArgs->kmerSize, bestKmerLocations);
    }
    printf("-------------------------------\n\n\n");
}