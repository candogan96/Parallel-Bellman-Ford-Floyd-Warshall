/******************************************************************************
 *
 * omp-floyd-warshall.c - All-source shortest paths
 *
 * Written in 2018 by Can Dogan <can.dogan(at)studio.unibo.it>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * This program computes all-source shortest paths on directed
 * graphs using Floyd-Warshall algorithm
 *
 * To compile this program:
 * gcc -fopenmp -std=c99 -Wall -Wpedantic omp-floyd-warshall.c -o omp-floyd-warshall -lm
 *
 * To compute the distances using the graph rome99.gr:
 * ./omp-floyd-warshall rome99.gr rome99.dist
 *
 * To print a specific path from source 0 to destination 3352 in path.txt file:
 * ./omp-floyd-warshall rome99.gr rome99.dist 0 3352 path.txt
 *
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> /* for isinf(), fminf() and HUGE_VAL */
#include <assert.h>
#include <omp.h>

typedef struct {
    int src, dst;
    float w;
} edge_t;

typedef struct {
    int n; /* number of nodes */
    int m; /* length of the edges array */
    edge_t *edges; /* array of edges */
} graph_t;

int cmp_edges(const void* p1, const void* p2)
{
    edge_t *e1 = (edge_t*)p1;
    edge_t *e2 = (edge_t*)p2;
    return (e1->dst - e2->dst);
}

void sort_edges(graph_t *g)
{
    qsort(g->edges, g->m, sizeof(edge_t), cmp_edges);
}


/**
 * Load a graph description in DIMACS format from file |f|; store the
 * graph in |g| (the caller is responsible for providing a pointer to
 * an already allocated object). This function interprets |g| as an
 * _undirected_ graph; this means that each edge (u,v) in the input
 * file appears twice in |g|, as (u,v) and (v,u)
 * respectively. Therefore, if the input graph has m edges, the edge
 * array of |g| will have 2*m elements. For more information on the
 * DIMACS challenge format see
 * http://www.diag.uniroma1.it/challenge9/index.shtml
 */
void load_dimacs(FILE *f, graph_t* g)
{
    const size_t buflen = 1024;
    char buf[buflen], prob[buflen];
    int n, m, src, dst, w;
    int cnt = 0; /* edge counter */
    int idx = 0; /* index in the edge array */
    int nmatch;

    while ( fgets(buf, buflen, f) ) {
        switch( buf[0] ) {
        case 'c':
            break; /* ignore comment lines */
        case 'p':
            /* Parse problem format; expect type "shortest path" */
            sscanf(buf, "%*c %s %d %d", prob, &n, &m);
            if (strcmp(prob, "sp")) {
                fprintf(stderr, "FATAL: unknown DIMACS problem type %s\n", prob);
                exit(-1);
            }
            fprintf(stderr, "DIMACS %s with %d nodes and %d edges\n", prob, n, m);
            g->n = n;
            g->m = 2*m;
            g->edges = (edge_t*)malloc((g->m)*sizeof(edge_t)); assert(g->edges);
            cnt = idx = 0;
            break;
        case 'a':
            nmatch = sscanf(buf, "%*c %d %d %d", &src, &dst, &w);
            if (nmatch != 3) {
                fprintf(stderr, "FATAL: Malformed line \"%s\"\n", buf);
                exit(-1);
            }
            /* In the DIMACS format, nodes are numbered starting from
               1; we use zero-based indexing internally, so we
               decrement the ids by one */

            /* For each edge (u,v,w) in the input file, we insert two
               edges (u,v,w) and (v,u,w) in the edge array, one edge
               for each direction */
            g->edges[idx].src = src-1;
            g->edges[idx].dst = dst-1;
            g->edges[idx].w = w;
            idx++;
            g->edges[idx].src = dst-1;
            g->edges[idx].dst = src-1;
            g->edges[idx].w = w;
            idx++;
            cnt++;
            break;
        default:
            fprintf(stderr, "FATAL: unrecognized character %c in line \"%s\"\n", buf[0], buf);
            exit(-1);
        }
    }
    assert( 2*cnt == g->m );
    sort_edges(g);
}

float *IDX_f(float *d, int n, int i, int j)
{
    return d + i*n + j;
}

int *IDX_i(int *p, int n, int i, int j)
{
    return p + i*n + j;
}


/* Compute shortest paths between all nodes using Floyd-Warshall
   algorithm. This is a simple serial algorithm.
   g: the graph structure
   d: pointer of distances array
   p: pointer of predecessors array
   */
void floydwarshall(const graph_t* g, float *d, int *p)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, niter;

    // set all distances to INFINITY
    for (i=0; i<n; i++) {
        for(j=0; j<n; j++)
            *IDX_f(d, n, i, j) = INFINITY;
    }
    // set distances between a node and self to zero
    for(i=0; i<n; i++){
      *IDX_f(d, n, i, i) = 0.0f;
      *IDX_i(p, n, i, i) = i;
    }

    // set initial distances to edges weights
    int src, dst;
    float w;
    for(i=0; i<m; i++) {
        src = g->edges[i].src;
        dst = g->edges[i].dst;
        w = g->edges[i].w;

        *IDX_f(d, n, src, dst) = w;
        *IDX_i(p, n, src, dst) = src;
    }
/*
    printf("printing d\n\n");
    for(i=0; i<n; i++) {
      for(j=0; j<n; j++){
        printf("%f ", *IDX_f(d, n, i, j));
      }
      printf("\n");
    }
*/

    // start algorythm
    for(niter=0; niter<n; niter++){
      if (niter%100 == 0) {
        printf(".");
        fflush(stdout);
      }
      float d1, d2;
      float *d_i, *d_niter = d + niter * n;
      int *p_i;

#pragma omp parallel for default(none) shared(d, p, niter, d_niter) private(d1, d2, i, j, d_i, p_i)
      for(i=0; i<n; i++) {
        d_i = d + i * n;
        p_i = p + i * n;
        d1 = *(d_i + niter);
        for (j=0; j<n; j++) {

            if (isinf(d1))
              break;

            d2 = *(d_niter + j);

            if (d1+d2 < *(d_i + j) ) {
              *(d_i + j) = d1+d2;
              *(p_i + j) = *IDX_i(p, n, niter, j);
            }
        }
      }
    }
    fprintf(stderr, "\nfloydwarshall_parallel: %d iterations\n", niter);
}

void floydwarshall_serial(const graph_t* g, float *d, int *p)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, niter;

    // set all distances to INFINITY
    for (i=0; i<n; i++) {
        for(j=0; j<n; j++)
            *IDX_f(d, n, i, j) = INFINITY;
    }
    // set distances between a node and self to zero
    for(i=0; i<n; i++){
      *IDX_f(d, n, i, i) = 0.0f;
      *IDX_i(p, n, i, i) = i;
    }

    // set initial distances to edges weights
    int src, dst;
    float w;
    for(i=0; i<m; i++) {
        src = g->edges[i].src;
        dst = g->edges[i].dst;
        w = g->edges[i].w;

        *IDX_f(d, n, src, dst) = w;
        *IDX_i(p, n, src, dst) = src;
    }

    // start algorythm
    for(niter=0; niter<n; niter++){
      if (niter%100 == 0) {
        printf(".");
        fflush(stdout);
      }
      float d1, d2;
      float *d_i, *d_niter = d + niter * n;
      int *p_i;

      for(i=0; i<n; i++) {
        d_i = d + i * n;
        p_i = p + i * n;
        d1 = *(d_i + niter);
        for (j=0; j<n; j++) {

            if (isinf(d1))
              break;

            d2 = *(d_niter + j);

            if (d1+d2 < *(d_i + j) ) {
              *(d_i + j) = d1+d2;
              *(p_i + j) = *IDX_i(p, n, niter, j);
            }
        }
      }
    }
    fprintf(stderr, "\nfloydwarshall_serial: %d iterations\n", niter);
}


/* Check distances. Return 0 if d1 and d2 contain the same values (up
   to a given tolerance), -1 otherwise. */
int checkdist( float *d1, float *d2, int n)
{
    const float epsilon = 1e-5;
    int i, j;
    for (i=0; i<n; i++) {
      for(j=0; j<n; j++)
        if ( fabsf(*IDX_f(d1, n, i, j) - *IDX_f(d2, n, i, j)) > epsilon ) {
            fprintf(stderr, "FATAL: d1[%d][%d]=%f, d2[%d][%d]=%f\n", i, j, *IDX_f(d1, n, i, j), i, j, *IDX_f(d2, n, i, j));
            return -1;
        }
    }
    return 0;
}


void printPath(FILE *f, int *p, int n, int src, int dst) {
    if (p[src * n + dst] != src)
        printPath(f, p, n, src, p[src * n + dst]);

    fprintf(f, "%d\n", dst);
}


int main( int argc, char* argv[] )
{
    graph_t g;
    int i, j, src = -1, dst = -1;
    float *d_serial, *d_parallel;
    int *p_serial, *p_parallel;
    float tstart, t_serial, t_parallel;

    const char *infile, *outfile, *path_outfile;
    FILE *in, *out, *pathout;
    if (argc == 6) {
        src = atoi(argv[3]);
        dst = atoi(argv[4]);
        path_outfile = argv[5];
    } else if ( argc != 3 ) {
        fprintf(stderr, "Usage: %s infile outfile (src) (dst) (path_outfile)\n", argv[0]);
        return -1;
    }
    infile = argv[1];
    outfile = argv[2];

    in = fopen(infile, "r");
    if (in == NULL) {
        fprintf(stderr, "FATAL: can not open \"%s\" for reading\n", infile);
        exit(-1);
    }


    /* required by atomicRelax() */
    assert( sizeof(float) == sizeof(int) );

    load_dimacs(in, &g);
    fclose(in);

    const size_t sz = (g.n * g.n) * sizeof(*d_serial);

    d_serial = (float*)malloc(sz); assert(d_serial);
    d_parallel = (float*)malloc(sz); assert(d_parallel);
    p_serial = (int*)malloc(g.n * g.n * sizeof(int)); assert(p_serial);
    p_parallel = (int*)malloc(g.n * g.n * sizeof(int)); assert(p_parallel);
/*
    if (src < 0 || src >= g.n || dst >= g.n) {
        fprintf(stderr, "FATAL: invalid source or destination node (should be within %d-%d)\n", 0, g.n-1);
        exit(-1);
    }
*/
    tstart = omp_get_wtime();
    floydwarshall_serial(&g, d_serial, p_serial);
    t_serial = omp_get_wtime() - tstart;
    fprintf(stderr, "Serial execution time....... %f\n", t_serial);

    tstart = omp_get_wtime();
    floydwarshall(&g, d_serial, p_serial);
    t_parallel = omp_get_wtime() - tstart;
    fprintf(stderr, "Parallel execution time....... %f (%.2fx)\n", t_parallel, t_serial/t_parallel);


    out = fopen(outfile, "w");
    if ( out == NULL ) {
        fprintf(stderr, "FATAL: can not open \"%s\" for writing", outfile);
        exit(-1);
    }

    /* print distances to output file */
    for (i=0; i<g.n; i++) {
      for(j=0; j<g.n; j++)
        fprintf(out, "d %d %d %f\n", i, j, *IDX_f(d_serial, g.n, i, j));
    }
    fclose(out);
    /* print path to file if destination parameter is set */

/*
    printf("printing p:\n");
    for(i=0; i<g.n; i++) {
      for(j=0; j<g.n; j++) {
        printf("%d ", *IDX_i(p_serial, g.n, i, j));
      }
      printf("\n");
    }
*/

    if (path_outfile != NULL && src >= 0 && dst >= 0) {
      pathout = fopen(path_outfile, "w");
      if ( pathout == NULL ) {
          fprintf(stderr, "FATAL: can not open \"%s\" for writing", path_outfile);
          exit(-1);
      }
      printPath(pathout, p_serial, g.n, src, dst);
      fclose(pathout);
    }
    free(d_serial);
    free(d_parallel);
    free(p_serial);
    free(p_parallel);

    return 0;
}
