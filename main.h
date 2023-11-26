#ifndef KMEANS_H
#define KMEANS_H

// Constants
const int THREADSPERBLOCK = 512;
const int NUM_SYMBOLS = 4; // number of nucleotides

// Function headers
void parallel_generateProfileMatrix(char *d_data, int *d_profile, 
                                    int *d_kmerLocations, int seqToIgnore, 
                                    int numSequences, int seqLen, int kmerSize);

void parallel_generateProfileProbabilities(char *d_data, int *d_profile, 
                                           int *d_profileGeneratedProbabilities, 
                                           int numProfileProbabilities, int seqToIgnore,
                                           int seqLen, int kmerSize);
                            
int parallel_calculateKmerScore(char *d_data, int *d_kmerLocations, int *d_kmerSubscores, 
                                 int numSequences, int seqLen, int kmerSize);

#endif