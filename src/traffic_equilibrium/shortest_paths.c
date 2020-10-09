#include "igraph_dqueue.h"

static int igraph_get_shortest_paths_bellman_ford(
	const igraph_t *graph,
	igraph_vector_ptr_t *edges, // paths from from to each of to
	igraph_vector_t *path_costs, // cost of each path
	igraph_integer_t from,
	igraph_vs_t to,
	const igraph_vector_t *weights,
	igraph_neimode_t mode
	)
{
	/* Implementation details:
       - `parents` assigns the inbound edge IDs of all vertices in the
         shortest path tree to the vertices. In this implementation, the
         edge ID + 1 is stored, zero means unreachable vertices.
    */
	long int no_of_nodes = igraph_vcount(graph);
	long int no_of_edges = igraph_ecount(graph);
	igraph_lazy_inclist_t inclist;
	long int i, j, k, head;
	long int source = (long int) from;
	long int no_of_from, no_of_to;

	igraph_dqueue_t Q, Q_pape;
	igraph_vector_t clean_vertices;
	igraph_vector_t num_queued;
	igraph_vit_t tovit;
	igraph_real_t my_infinity = IGRAPH_INFINITY;
	igraph_bool_t all_to;
	igraph_vector_t dist;
	igraph_real_t tmp_dist;

	long int nlen, nei, target;
	long int *parents;

	/* setup parents array */
    parents = igraph_Calloc(no_of_nodes, long int);
    if (parents == 0) {
        IGRAPH_ERROR("Can't calculate shortest paths", IGRAPH_ENOMEM);
    }
    // IGRAPH_FINALLY(igraph_free, parents);

    if (igraph_vector_size(weights) != no_of_edges) {
        IGRAPH_ERROR("Weight vector length does not match", IGRAPH_EINVAL);
    }

    IGRAPH_DQUEUE_INIT_FINALLY(&Q, no_of_nodes);
    IGRAPH_DQUEUE_INIT_FINALLY(&Q_pape, no_of_nodes);
    IGRAPH_VECTOR_INIT_FINALLY(&clean_vertices, no_of_nodes);
    IGRAPH_VECTOR_INIT_FINALLY(&num_queued, no_of_nodes);
    IGRAPH_CHECK(igraph_lazy_inclist_init(graph, &inclist, mode));
    IGRAPH_FINALLY(igraph_lazy_inclist_destroy, &inclist);

	IGRAPH_CHECK(igraph_vit_create(graph, to, &tovit));
	// Don't register tovit for de-allocation; do it manually
	// IGRAPH_FINALLY(igraph_vit_destroy, &tovit);
	no_of_to = IGRAPH_VIT_SIZE(tovit);

    IGRAPH_VECTOR_INIT_FINALLY(&dist, no_of_nodes);

    /* bellman-ford */
    igraph_vector_fill(&dist, my_infinity);
    VECTOR(dist)[source] = 0;
	// igraph_vector_null(&clean_vertices);
	// mark all as vertices as initially clean
	igraph_vector_fill(&clean_vertices, 1.0);
	igraph_vector_null(&num_queued);

	/* Fill the queue with vertices to be checked
	for (j = 0; j < no_of_nodes; j++) {
		IGRAPH_CHECK(igraph_dqueue_push(&Q, j));
	}
	*/
	IGRAPH_CHECK(igraph_dqueue_push(&Q, source));

	/* Run shortest paths */
	while (!igraph_dqueue_empty(&Q) || !igraph_dqueue_empty(&Q_pape)) {
		igraph_vector_t *neis;

		// pop from pape's queue if non-empty
		if (!igraph_dqueue_empty(&Q_pape)) {
			j = (long int) igraph_dqueue_pop_back(&Q_pape);
		} else { // pop from the regular queue
			j = (long int) igraph_dqueue_pop(&Q);
		}
		//printf("popped node %li...", j);

		VECTOR(clean_vertices)[j] = 1;
		VECTOR(num_queued)[j] += 1;

		/*
		if (VECTOR(num_queued)[j] > no_of_nodes) {
			IGRAPH_ERROR("cannot run Bellman-Ford algorithm", IGRAPH_ENEGLOOP);
		}
		*/


		/* If we cannot get to j in finite time yet, there is no need to relax
		 * its edges */
		if (!IGRAPH_FINITE(VECTOR(dist)[j])) {
			continue;
		}

		neis = igraph_lazy_inclist_get(&inclist, (igraph_integer_t) j);
		nlen = igraph_vector_size(neis);

		for (k = 0; k < nlen; k++) {
			nei = (long int) VECTOR(*neis)[k];
			target = IGRAPH_OTHER(graph, nei, j);
			tmp_dist = VECTOR(dist)[j] + VECTOR(*weights)[nei];
			if (VECTOR(dist)[target] > tmp_dist) {
				/* relax the edge */
				//printf("relaxing edge %li->%li...", j, target);
				VECTOR(dist)[target] = tmp_dist;
				/* update parent of target */
                parents[target] = nei + 1;
				if (VECTOR(clean_vertices)[target]) {
					//printf("pushing node %li...", target);
					VECTOR(clean_vertices)[target] = 0;
					/* Vanilla Bellman-Ford
					IGRAPH_CHECK(igraph_dqueue_push(&Q, target));
					 */

					/* Pape's rule: if the target has been scanned already,
					 * add it to pape's queue to ensure it is drawn first
					 * otherwise add it to the regular queue.
					if (VECTOR(num_queued)[target] > 0) {
						IGRAPH_CHECK(igraph_dqueue_push(&Q_pape, target));
					} else {
						IGRAPH_CHECK(igraph_dqueue_push(&Q, target));
				    }
					 */

				    /* SLF
				     * Compare the label of the target node with the head of Q.
				     * If the target label is smaller, append to pape's queue
				     * else append to regular queue.
				    */
				    if (!igraph_dqueue_empty(&Q_pape)) {
				    	head = igraph_dqueue_back(&Q_pape);
				    } else {
				    	head = igraph_dqueue_head(&Q);
				    }
				    if (tmp_dist <= VECTOR(dist)[head]) {
						IGRAPH_CHECK(igraph_dqueue_push(&Q_pape, target));
				    } else {
						IGRAPH_CHECK(igraph_dqueue_push(&Q, target));
				    }

				}
			}
		}
		//printf("\n");
	}

	/* populate edges and path costs */
	if (edges) {
        for (IGRAPH_VIT_RESET(tovit), i = 0; !IGRAPH_VIT_END(tovit); IGRAPH_VIT_NEXT(tovit), i++) {
        	long int node = IGRAPH_VIT_GET(tovit);
        	long int size, act, edge;

        	igraph_vector_t *evec = 0;
        	evec = VECTOR(*edges)[i];
        	igraph_vector_clear(evec);

			if (path_costs) {
				VECTOR(*path_costs)[i] = VECTOR(dist)[node];
			}

        	size = 0;
        	act = node;
        	while (parents[act]) {
        		size++;
        		edge = parents[act] - 1;
        		act = IGRAPH_OTHER(graph, edge, act);
        	}
			IGRAPH_CHECK(igraph_vector_resize(evec, size));

            act = node;
            while (parents[act]) {
                edge = parents[act] - 1;
                act = IGRAPH_OTHER(graph, edge, act);
                size--;
				VECTOR(*evec)[size] = edge;
            }

        }
	}

	/* de-allocate */
	igraph_vit_destroy(&tovit);

	igraph_vector_destroy(&dist);
    igraph_dqueue_destroy(&Q);
    igraph_dqueue_destroy(&Q_pape);
    igraph_vector_destroy(&clean_vertices);
    igraph_vector_destroy(&num_queued);
    igraph_lazy_inclist_destroy(&inclist);
    IGRAPH_FINALLY_CLEAN(6);

    igraph_Free(parents);

	return 0;
}
