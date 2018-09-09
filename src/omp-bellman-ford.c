/******************************************************************************
 *
 * omp-bellman-ford.c - Single-source shortest paths
 *
 * Written in 2017, 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Edited in 2018 by Can Dogan <can.dogan(at)studio.unibo.it>
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
 * This program computes single-source shortest paths on directed
 * graphs using Bellman-Ford algorithm
 *
 * To compile this program:
 * gcc -fopenmp -std=c99 -Wall -Wpedantic omp-bellman-ford.c -o omp-bellman-ford -lm
 *
 * To compute the distances using the graph rome99.gr, using node 0 as
 * the source:
 * ./omp-bellman-ford 0 rome99.gr rome99.dist
 *
 * To print the path from source 0 to destination 3352 in path.txt file:
 * ./omp-bellman-ford 0 rome99.gr rome99.dist 3352 path.txt
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

/* Set *v = min(*v, x), atomically; return 1 if the value of *v changed */
static inline int atomicRelax(float *v, float x)
{
    union {
        float vf;
        int vi;
    } oldval, newval;

    int* volatile int_v = (int*)v;

    if ( *v < x )
        return 0;

    do {
        oldval.vf = *v;
        newval.vf = (oldval.vf < x ? oldval.vf : x);
    } while (! __sync_bool_compare_and_swap(int_v, oldval.vi, newval.vi) );
    return (newval.vi != oldval.vi);
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

/* Compute distances from source node |s| using Dijkstra's algorithm.
   This implementation is extremely inefficient since it traverses the
   whole edge list at each iteration, while a smarter implementation
   would traverse only the edges incident to the frontier of the
   shortest path tree. However, it is quite easy to parallelize, and
   this is what counts here. |g| is the input graph; |s| is the source
   node id; |d| is the array of distances, that must be pre-allocated
   by the caller to hold |g->n| elements. When the function
   terminates, d[i] is the distance of node i from node s. */
void dijkstra(const graph_t* g, int s, float *d)
{
    const int n = g->n;
    const int m = g->m;
    float best_dist; /* minimum distance */
    int best_node; /* node with minimum distance */
    int i;

    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0;
    do {
        best_dist = INFINITY;
        best_node = -1;
#pragma omp parallel default(none) shared(g,d,best_dist,best_node)
        {
            const int num_threads = omp_get_max_threads();
            const int my_id = omp_get_thread_num();
            const int jstart = m*my_id/num_threads;
            const int jend = m*(my_id+1)/num_threads;
            float my_best_dist = INFINITY;
            int my_best_node = -1;
            int j;

            for (j=jstart; j<jend; j++) {
                const int src = g->edges[j].src;
                const int dst = g->edges[j].dst;
                const float dist = d[src] + g->edges[j].w;
                if ( isfinite(d[src]) && isinf(d[dst]) && (dist < my_best_dist) ) {
                    my_best_dist = dist;
                    my_best_node = dst;
                }
            }
#pragma omp critical
            {
                if ( my_best_dist < best_dist ) {
                    best_dist = my_best_dist;
                    best_node = my_best_node;
                }
            }
        }
        if ( isfinite(best_dist) ) {
            assert( best_node >= 0 );
            d[best_node] = best_dist;
        }
    } while (isfinite(best_dist));
}

/* Compute shortest paths from node s using Bellman-Ford
   algorithm. This is a simple serial algorithm (probably the simplest
   algorithm to compute shortest paths), that might be used to check
   the correctness of function dijkstra(). This version is slightly
   optimized with respect to the "canonical" implementation of the
   Bellman-Ford algorithm: if no distance is updated after a
   relaxation phase, this function terminates immediately since no
   distances will be updated in future iterations.
   g: the graph structure
   s: source node id
   d: pointer of distances array
   p: pointer of predecessors array
   */
void bellmanford(const graph_t* g, int s, float *d, int *p)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, updated, niter;

    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0;
    for(niter=0; niter<n; niter++) {
        updated = 0;
        for (j=0; j<m; j++) {
            const int src = g->edges[j].src;
            const int dst = g->edges[j].dst;
            const float w = g->edges[j].w;
            if ( d[src] + w < d[dst] ) {
                d[dst] = d[src] + w;
                p[dst] = src;
                updated = 1;
            }
        }

        if (0 == updated) {
            break;
        }
    }

    /* Another loop to check negative cycles */
    for (j=0; j<m; j++) {
        const int src = g->edges[j].src;
        const int dst = g->edges[j].dst;
        const float w = g->edges[j].w;
        if ( d[src] + w < d[dst] ) {
            fprintf(stderr, "Error: graph contains negative cycles.\n");
        }
    }
    fprintf(stderr, "bellmanford: %d iterations\n", niter);
}

/* Compute shortest paths from node s using Bellman-Ford
   algorithm. This is a parallel implementation making use of mutex locks
   provided by OpenMP. This version is slightly
   optimized with respect to the "canonical" implementation of the
   Bellman-Ford algorithm: if no distance is updated after a
   relaxation phase, this function terminates immediately since no
   distances will be updated in future iterations.
   g: the graph structure
   s: source node id
   d: pointer of distances array
   p: pointer of predecessors array
   */
void bellmanford_atomic(const graph_t* g, int s, float *d, int* p, omp_lock_t *locks) {
    const int n = g->n;
    const int m = g->m;
    int i, j, updated, niter = 0;

    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0.0f;

    for(niter=0; niter<n; niter++) {
        updated = 0;
#pragma omp parallel for default(none) shared(g, d, p, locks) reduction(|:updated)
        for (j=0; j<m; j++) {
            const int src = g->edges[j].src;
            const int dst = g->edges[j].dst;
            const float w = g->edges[j].w;

            if (!isinf(d[src]) && !isinf(w) && d[src] + w < d[dst]) {
                int first = src < dst ? src : dst;
                int second = src > dst ? src : dst;
                omp_set_lock(&locks[first]);
                omp_set_lock(&locks[second]);
                float d_new = d[src] + w;
                if (d_new < d[dst]) {
                    d[dst] = d_new;
                    p[dst] = src;
                    updated |= 1;
                }
                omp_unset_lock(&locks[second]);
                omp_unset_lock(&locks[first]);
            }
        }


        if (0 == updated) {
          break;
        }
    }

    /* Another loop to check negative cycles */
    for (j=0; j<m; j++) {
        const int src = g->edges[j].src;
        const int dst = g->edges[j].dst;
        const float w = g->edges[j].w;
        if ( d[src] + w < d[dst] ) {
            fprintf(stderr, "Error: graph contains negative cycles.\n");
            break;
        }
    }
    fprintf(stderr, "bellmanford_atomic: %d iterations\n", niter);
}

void bellmanford_atomic_inlined(const graph_t* g, int s, float *d)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, updated, niter = 0;

    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0.0f;
    for(niter=0; niter<n; niter++) {
        updated = 0;
#pragma omp parallel for default(none) shared(g, d) reduction(|:updated)
        for (j=0; j<m; j++) {
            const int src = g->edges[j].src;
            const int dst = g->edges[j].dst;
            const float w = g->edges[j].w;

            if ( d[src] + w < d[dst] ) {
                union {
                    float vf;
                    int vi;
                } oldval, newval;

                volatile int* dist_p = (int*)(d + dst);

                do {
                    oldval.vf = d[dst];
                    newval.vf = fminf(d[src]+w, d[dst]);
                } while (! __sync_bool_compare_and_swap(dist_p, oldval.vi, newval.vi) );
                updated |= (newval.vi != oldval.vi);
            }
        }
        if (0 == updated) {
          break;
        }
    }

    /* Another loop to check negative cycles */
    for (j=0; j<m; j++) {
        const int src = g->edges[j].src;
        const int dst = g->edges[j].dst;
        const float w = g->edges[j].w;
        if ( d[src] + w < d[dst] ) {
            fprintf(stderr, "Error: graph contains negative cycles.\n");
        }
    }
    fprintf(stderr, "bellmanford_atomic: %d iterations\n", niter);
}

/* Bellman-Ford algorithm without syncronization. Note that this
   implementation is technically NOT CORRECT, since multiple OpenMP
   threads might "relax" the same distance at the same time, resulting
   in a race condition. However, for some reasons that I do not
   understand, the program seems to always compute the correct
   distance on the test cases I tried. */
void bellmanford_none(const graph_t* g, int s, float *d, int *p)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, updated = 0, niter = 0;

    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0.0f;
    for(niter=0; niter<n; niter++) {
        updated = 0;
#pragma omp parallel for default(none) shared(g, d, p) reduction(|:updated)
        for (j=0; j<m; j++) {
            const int src = g->edges[j].src;
            const int dst = g->edges[j].dst;
            const float w = g->edges[j].w;
            if ( d[src]+w < d[dst] ) {
                updated |= 1;
                d[dst] = d[src]+w;
                p[dst] = src;
            }
        }
      if (0 == updated) {
        break;
      }
    }

    /* Another loop to check negative cycles */
    for (j=0; j<m; j++) {
        const int src = g->edges[j].src;
        const int dst = g->edges[j].dst;
        const float w = g->edges[j].w;
        if ( d[src] + w < d[dst] ) {
            fprintf(stderr, "Error: graph contains negative cycles.\n");
        }
    }
    fprintf(stderr, "belmanford_none: %d iterations\n", niter);
}

/* Check distances. Return 0 if d1 and d2 contain the same values (up
   to a given tolerance), -1 otherwise. */
int checkdist( float *d1, float *d2, int n)
{
    const float epsilon = 1e-5;
    int i;
    for (i=0; i<n; i++) {
        if ( fabsf(d1[i] - d2[i]) > epsilon ) {
            fprintf(stderr, "FATAL: d1[%d]=%f, d2[%d]=%f\n", i, d1[i], i, d2[i]);
            return -1;
        }
    }
    return 0;
}

void printPath(FILE *f, int *p, int src, int dst) {
    if (p[dst] != src)
        printPath(f, p, src, p[dst]);

    fprintf(f, "%d\n", dst);
}

int main( int argc, char* argv[] )
{
    graph_t g;
    int i, src = 0, dst = -1;
    float *d_serial, *d_atomic, *d_none;
    int *p_serial, *p_atomic, *p_none;
    float tstart, t_serial, t_atomic, t_none;

    const char *infile, *outfile, *pathoutfile;
    FILE *in, *out, *pathout;
    if (argc == 6) {
        dst = atoi(argv[4]);
        pathoutfile = argv[5];
    } else if ( argc != 4 ) {
        fprintf(stderr, "Usage: %s source_node infile outfile (dst_node) (path_outfile)\n", argv[0]);
        return -1;
    }
    src = atoi(argv[1]);
    infile = argv[2];
    outfile = argv[3];

    in = fopen(infile, "r");
    if (in == NULL) {
        fprintf(stderr, "FATAL: can not open \"%s\" for reading\n", infile);
        exit(-1);
    }


    /* required by atomicRelax() */
    assert( sizeof(float) == sizeof(int) );

    load_dimacs(in, &g);
    fclose(in);

    const size_t sz = (g.n) * sizeof(*d_serial);

    d_serial = (float*)malloc(sz); assert(d_serial);
    d_atomic = (float*)malloc(sz); assert(d_atomic);
    d_none = (float*)malloc(sz); assert(d_none);

    p_serial = (int*)malloc(g.n * sizeof(int)); assert(p_serial);
    p_atomic = (int*)malloc(g.n * sizeof(int)); assert(p_atomic);
    p_none = (int*)malloc(g.n * sizeof(int)); assert(p_none);

      if (src < 0 || src >= g.n || dst >= g.n) {
          fprintf(stderr, "FATAL: invalid source or destination node (should be within %d-%d)\n", 0, g.n-1);
          exit(-1);
      }

    tstart = omp_get_wtime();
    bellmanford(&g, src, d_serial, p_serial);
    t_serial = omp_get_wtime() - tstart;
    fprintf(stderr, "Serial execution time....... %f\n", t_serial);

    tstart = omp_get_wtime();
    bellmanford_none(&g, src, d_none, p_none);
    t_none = omp_get_wtime() - tstart;
    fprintf(stderr, "Par. exec. time (no sync.).. %f (%.2fx)\n", t_none, t_serial/t_none);
    checkdist(d_serial, d_none, g.n);

    omp_lock_t *locks;
    locks = (omp_lock_t*)malloc(g.n * sizeof(omp_lock_t)); assert(locks);

    for (i=0; i< g.n; i++)
        omp_init_lock(&locks[i]);

    tstart = omp_get_wtime();
    bellmanford_atomic(&g, src, d_atomic, p_atomic, locks);
    t_atomic = omp_get_wtime() - tstart;
    fprintf(stderr, "Par. exec. time (atomic).... %f (%.2fx)\n", t_atomic, t_serial/t_atomic);
    checkdist(d_serial, d_atomic, g.n);

    out = fopen(outfile, "w");
    if ( out == NULL ) {
        fprintf(stderr, "FATAL: can not open \"%s\" for writing", outfile);
        exit(-1);
    }

    /* print distances to output file */
    for (i=0; i<g.n; i++) {
        fprintf(out, "d %d %d %f\n", src, i, d_serial[i]);
    }
    fclose(out);
    /* print path to file if destination parameter is set */
    if (pathoutfile != NULL && dst >= 0) {
      pathout = fopen(pathoutfile, "w");
      if ( pathout == NULL ) {
          fprintf(stderr, "FATAL: can not open \"%s\" for writing", pathoutfile);
          exit(-1);
      }
      printPath(pathout, p_serial, src, dst);
      fclose(pathout);
    }
    return 0;
}
