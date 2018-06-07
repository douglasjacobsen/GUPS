#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "immintrin.h"
#include "omp.h"

int64_t* table;
const int64_t poly = 0x0000000000000007UL;

__declspec(noinline) void perform_updates( const int64_t scalar, const int64_t table_size, const int64_t nupdates ) {/*{{{*/
    int64_t i;
    int location = 10294 % table_size;

    #pragma omp parallel for private(location)
    for ( i = 0; i < nupdates; i++ ) {
        location = ( (location + ( location + 1 ) ) ^ ( poly & ( location >> 31 ) ) ) % table_size ;

        table[location] = scalar;
    }
}/*}}}*/


int main(int argc, char** argv) {

    int alignment, ntrials, nthreads;
    int64_t bits, N, i, nups;
    char* stopstring;

    if ( argc != 4 ) {
        printf("Usage Statement:\n");
        printf("    GUPS.x [bits] [nups] [trials]\n");
        printf("        bits = Hash table size 2<<bits.\n");
        printf("        nups = Number of updates to perform.\n");
        printf("        trials = Number of trials. A trial is nups updates of the hash table.\n");
        return(1);
    }

    bits = strtoul(argv[1], &stopstring, 10);
    nups = strtoul(argv[2], &stopstring, 10);
    ntrials = strtoul(argv[3], &stopstring, 10);

    nthreads = 1;
#pragma omp parallel shared(nthreads)
    {
#pragma omp master
    {
        nthreads = omp_get_num_threads();
    }
    }

    N = 2<<bits;
    alignment = 2 * 1024 * 1024;

    printf("  ----- GUPS -----\n");
    printf("    bits = %u\n", bits);
    printf("    N = %u\n", N);
    printf("    nups = %u\n", nups);
    printf("    ntrials = %d\n", ntrials);
    printf("    nthreads = %d\n", nthreads);
    printf("    alignment = %d\n", alignment);

    table = (int64_t*)_mm_malloc( sizeof(int64_t) * N, alignment );

    for (i = 0; i < N; i++ ) {
        table[i] = 0;
    }

    // Warmup iterations
    for ( i = 0; i < 10; i++ ) {
        perform_updates(i, N, nups);
    }

    double start = omp_get_wtime();
    for ( i = 0; i < ntrials; i++ ) {
        perform_updates(i, N, nups);
    }
    double end = omp_get_wtime();

    printf( "Total time: %lf\n", end - start );
    printf( "GUPS: %lf\n", ((double)nups * 1e-9) / ( (end - start) / ntrials ));

    _mm_free( table );
}
