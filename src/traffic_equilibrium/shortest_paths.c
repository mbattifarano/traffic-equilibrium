
static int get_shortest_paths_bellman_ford(
	const igraph_t *graph,
	igraph_vector_ptr_t *edges, // paths from from to each of to
	igraph_vector_t *path_costs, // cost of each path (optional)
	long int source,
	igraph_vs_t to,
	const igraph_vector_t *weights,
	igraph_neimode_t mode
	)
{
	/* Implementation details:
	- `parents` assigns the inbound edge IDs of all vertices in the
	 shortest path tree to the vertices. In this implementation, the
	 edge ID + 1 is stored, zero means unreachable vertices.
	- we don't use IGRAPH_INFINITY in the distance vector during the
	 computation, as IGRAPH_FINITE() might involve a function call
	 and we want to spare that. So we store distance+1.0 instead of
	 distance, and zero denotes infinity.
	*/
	long int no_of_nodes = igraph_vcount(graph);
	igraph_lazy_inclist_t inclist;
	long int i, j, k;//, head;

	igraph_dqueue_t Q, Q_pape;
	long int q_count = 1, pape_count = 0;
	igraph_vector_t unclean_vertices;
	igraph_vector_t num_queued;
	igraph_vit_t tovit;
	igraph_vector_t dist;
	igraph_real_t tmp_dist;

	igraph_vector_t *neis;
	long int nlen, nei, target;
	long int *parents, *path_length;
	igraph_real_t dist_to_j, dist_to_target;

	/* setup parents array */
	parents = igraph_Calloc(no_of_nodes, long int);
	// Store the path length (number of edges) to each node
	path_length = igraph_Calloc(no_of_nodes, long int);

    igraph_dqueue_init(&Q, no_of_nodes);
    igraph_dqueue_init(&Q_pape, no_of_nodes);
    igraph_vector_init(&unclean_vertices, no_of_nodes);
    igraph_vector_init(&num_queued, no_of_nodes);  // whether or not a node has been added to the queue
    igraph_lazy_inclist_init(graph, &inclist, mode);
	igraph_vit_create(graph, to, &tovit);
    igraph_vector_init(&dist, no_of_nodes);

    /* bellman-ford */
    VECTOR(dist)[source] = 1.0; // store distance + 1
	igraph_dqueue_push(&Q, source);
	VECTOR(num_queued)[source] = 1;

	/* Run shortest paths */
	while (q_count + pape_count) {
		// pop from pape's queue if non-empty
		if (pape_count) {
			j = (long int) igraph_dqueue_pop(&Q_pape);
			pape_count--;
		} else { // pop from the regular queue
			j = (long int) igraph_dqueue_pop(&Q);
			q_count--;
		}

		VECTOR(unclean_vertices)[j] = 0;
		dist_to_j = VECTOR(dist)[j]; // 0 means unreachable

		neis = igraph_lazy_inclist_get(&inclist, (igraph_integer_t) j);
		nlen = igraph_vector_size(neis);

		for (k = 0; k < nlen; k++) {
			nei = (long int) VECTOR(*neis)[k];
			target = IGRAPH_OTHER(graph, nei, j);
			dist_to_target = VECTOR(dist)[target];
			tmp_dist = dist_to_j + VECTOR(*weights)[nei];
			if ((tmp_dist < dist_to_target) || (!dist_to_target) ) {
				/* relax the edge */
				VECTOR(dist)[target] = tmp_dist;
				/* update parent of target */
                parents[target] = nei + 1;
                path_length[target] = path_length[j] + 1;
				if (!VECTOR(unclean_vertices)[target]) {
					VECTOR(unclean_vertices)[target] = 1;

					/* Vanilla Bellman-Ford
					IGRAPH_CHECK(igraph_dqueue_push(&Q, target));
					*/

					/* Pape's rule: if the target has been scanned already,
					 * add it to pape's queue to ensure it is drawn first
					 * otherwise add it to the regular queue.
					 */
					if (VECTOR(num_queued)[target]) {
						igraph_dqueue_push(&Q_pape, target);
						pape_count++;
					} else {
						igraph_dqueue_push(&Q, target);
						q_count++;
						VECTOR(num_queued)[target] = 1;
				    }

				    /* SLF
				     * Compare the label of the target node with the head of Q.
				     * If the target label is smaller, append to pape's queue
				     * else append to regular queue.
				    if (pape_count) {
				    	head = igraph_dqueue_back(&Q_pape);
				    } else {
				    	head = igraph_dqueue_head(&Q);
				    }
				    if ((tmp_dist < VECTOR(dist)[head]) || (!VECTOR(dist)[head])) {
						igraph_dqueue_push(&Q_pape, target);
						pape_count++;
				    } else {
						igraph_dqueue_push(&Q, target);
						q_count++;
				    }
				    */
				}
			}
		}
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
				VECTOR(*path_costs)[i] = VECTOR(dist)[node] - 1.0;
			}

        	size = path_length[node];
        	/*
        	act = node;
        	while (parents[act]) {
        		size++;
        		edge = parents[act] - 1;
        		act = IGRAPH_OTHER(graph, edge, act);
        	}
        	if (size != path_length[node]) {
        		printf("size (%li) is not equal to the computed path length (%li)\n", size, path_length[node]);
        		abort();
        	}
        	*/
			igraph_vector_resize(evec, size);

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
    igraph_vector_destroy(&unclean_vertices);
	igraph_vector_destroy(&dist);
    igraph_dqueue_destroy(&Q);
    igraph_dqueue_destroy(&Q_pape);
    igraph_vector_destroy(&num_queued);
    igraph_lazy_inclist_destroy(&inclist);
    igraph_Free(parents);
    igraph_Free(path_length);

	return 0;
}
