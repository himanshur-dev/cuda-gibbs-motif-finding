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
#include <cuda_runtime.h>
#include "main.h"

// helper function to allow threads to get the correct number of blocks
// given a fixed block size
int getNumBlocks(int numThings) {
    return (numThings + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
}


// clears the profile matrix by setting every entry to 1, for Laplace's Rule of Succession
__global__ void parallel_emptyProfileMatrix(int * d_profile, int kmerSize) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < kmerSize * NUM_SYMBOLS) {
        d_profile[index] = 1;
    }
}

// note that there should be a 1 entry in every cell of the profile matrix
// before calling this function
__global__ void parallel_fillProfileMatrix(char *d_data, int *d_profile, int *d_kmerLocations,
                                            int seqToIgnore, int numSequences, int seqLen, int kmerSize) {
    int totalBlockParameters = (blockIdx.x * blockDim.x) + threadIdx.x;
    int currSeqIdx = totalBlockParameters / kmerSize;
    int currNucleotideIdx = totalBlockParameters % kmerSize;
    if (currSeqIdx < numSequences && currSeqIdx != seqToIgnore) {
        int dataIndex = (currSeqIdx * seqLen) + d_kmerLocations[currSeqIdx] + currNucleotideIdx;
        char currNucleotide = d_data[dataIndex];
        switch(currNucleotide) {
            case 'A':
                atomicAdd(&d_profile[(currNucleotideIdx * NUM_SYMBOLS)], 1);
                break;
            case 'C':
                atomicAdd(&d_profile[(currNucleotideIdx * NUM_SYMBOLS)+1], 1);
                break;
            case 'G':
                atomicAdd(&d_profile[(currNucleotideIdx * NUM_SYMBOLS)+2], 1);
                break;
            case 'T':
                atomicAdd(&d_profile[(currNucleotideIdx * NUM_SYMBOLS)+3], 1);
                break;
            default:
                assert(false);
                break;
        }
    }
}

// wrapper to call CUDA kernels to generate profile matrix
void parallel_generateProfileMatrix(char *d_data, int *d_profile, 
                                    int *d_kmerLocations, int seqToIgnore, 
                                    int numSequences, int seqLen, int kmerSize) {
    cudaError_t cErr;
    
    // clear out profile matrix
    int numEmptyProfileBlocks = getNumBlocks(kmerSize * NUM_SYMBOLS);
    parallel_emptyProfileMatrix<<<numEmptyProfileBlocks, THREADSPERBLOCK>>>(d_profile, kmerSize);
    cErr = cudaGetLastError();
    assert(cErr == cudaSuccess);

    // calculate new profile matrix
    int numFillProfileBlocks = getNumBlocks(numSequences * kmerSize);
    parallel_fillProfileMatrix<<<numFillProfileBlocks, THREADSPERBLOCK>>>(d_data, d_profile, 
        d_kmerLocations, seqToIgnore, numSequences, seqLen, kmerSize);
    cErr = cudaGetLastError();
    assert(cErr == cudaSuccess);
}


__global__ void parallel_calculateProfileProbabilities(char *d_data, int *d_profile, 
                                                       int *d_profileGeneratedProbabilities, 
                                                       int numProfileProbabilities, int seqToIgnore,
                                                       int seqLen, int kmerSize) {
    char *seq = &(d_data[seqToIgnore * seqLen]);
    int seqIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (seqIdx < numProfileProbabilities) {
        int currProbability = 1;
        for (int nucleotideOffset = 0; nucleotideOffset < kmerSize; nucleotideOffset++) {
            int nucleotideIdx = seqIdx + nucleotideOffset;
            char currNucleotide = seq[nucleotideIdx];
            switch(currNucleotide) {
                case 'A':
                    currProbability *= d_profile[(nucleotideOffset * NUM_SYMBOLS)];
                    break;
                case 'C':
                    currProbability *= d_profile[(nucleotideOffset * NUM_SYMBOLS)+1];
                    break;
                case 'G':
                    currProbability *= d_profile[(nucleotideOffset * NUM_SYMBOLS)+2];
                    break;
                case 'T':
                    currProbability *= d_profile[(nucleotideOffset * NUM_SYMBOLS)+3];
                    break;
                default:
                    assert(false);
                    break;
            }
        }
        d_profileGeneratedProbabilities[seqIdx] = currProbability;
    }
}

// wrapper function
void parallel_generateProfileProbabilities(char *d_data, int *d_profile, 
                                                       int *d_profileGeneratedProbabilities, 
                                                       int numProfileProbabilities, int seqToIgnore,
                                                       int seqLen, int kmerSize) {
    cudaError_t cErr;

    // calculate the profile probabilities
    int numProfileProbabilityBlocks = getNumBlocks(numProfileProbabilities);
    parallel_calculateProfileProbabilities<<<numProfileProbabilityBlocks, THREADSPERBLOCK>>>(
        d_data, d_profile, d_profileGeneratedProbabilities, numProfileProbabilities, seqToIgnore,
        seqLen, kmerSize
    );
    cErr = cudaGetLastError();
    assert(cErr == cudaSuccess);
}