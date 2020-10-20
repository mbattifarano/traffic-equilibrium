typedef struct node_data {
	long int parent;    // previous node in the shortest path
	double distance;  // distance of the shortest path to this node
	long int length;  // number of edges in the shortest paths to this node
	int visited;      // 1 if this node has been queued before
	int dirty;      // 1 if this node was relabeled on the last iteration
} node_data_t;

static int get_shortest_paths_bellman_ford(
	const igraph_t *graph,
	igraph_vector_ptr_t *edges, // paths from from to each of to
	igraph_vector_t *path_costs, // cost of each path (optional)
	long int source,
	igraph_vs_t to,
	const igraph_vector_t *weights,
	igraph_neimode_t mode
	) {
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
	long int i, k, nlen, nei, size, act, edge;

	igraph_dqueue_t Q, Q_pape;
	long int q_count = 1, pape_count = 0;
	igraph_vit_t tovit;

	igraph_vector_t *neis;
	node_data_t *data = igraph_Calloc(no_of_nodes, node_data_t);

	long int u, v;
	node_data_t *u_data, *v_data;
	igraph_real_t dist_to_u, dist_to_v;
	igraph_real_t tmp_dist;

	igraph_dqueue_init(&Q, no_of_nodes);
	igraph_dqueue_init(&Q_pape, no_of_nodes);
	igraph_lazy_inclist_init(graph, &inclist, mode);
	igraph_vit_create(graph, to, &tovit);

	/* bellman-ford */
	igraph_dqueue_push(&Q, source);
	u_data = (data + source);
	u_data->visited = 1;
	u_data->distance = 1.0;

	/* Run shortest paths */
	while (q_count + pape_count) {
		// pop from pape's queue if non-empty
		if (pape_count) {
		  u = (long int) igraph_dqueue_pop(&Q_pape);
		  pape_count--;
		} else { // pop from the regular queue
		  u = (long int) igraph_dqueue_pop(&Q);
		  q_count--;
		}
		u_data = (data + u);

		u_data->dirty = 0; // set u to clean
		dist_to_u = u_data->distance; // 0 means unreachable

		neis = igraph_lazy_inclist_get(&inclist, (igraph_integer_t) u);
		nlen = igraph_vector_size(neis);

		for (k = 0; k < nlen; k++) {
			nei = (long int) VECTOR(*neis)[k];
			v = IGRAPH_OTHER(graph, nei, u);
			v_data = (data + v);
			dist_to_v = v_data->distance;
			tmp_dist = dist_to_u + VECTOR(*weights)[nei];
			if ((tmp_dist < dist_to_v) || (!dist_to_v) ) {
				/* relax the edge */
				v_data->distance = tmp_dist;
				v_data->parent = nei + 1;
				v_data->length = u_data->length + 1;

				if (!(v_data->dirty)) {
					v_data->dirty = 1;

					/* Vanilla Bellman-Ford
					igraph_dqueue_push(&Q, target);
					*/

					/* Pape's rule:
					* If the target has been scanned already,
					* add it to pape's queue to ensure it is drawn first
					* otherwise add it to the regular queue.
					*/
					if (v_data->visited) {
						igraph_dqueue_push(&Q_pape, v);
						pape_count++;
					} else {
						igraph_dqueue_push(&Q, v);
						q_count++;
						v_data->visited = 1;
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
			u = IGRAPH_VIT_GET(tovit);
			u_data = (data + u);

			igraph_vector_t *evec = 0;
			evec = VECTOR(*edges)[i];
			igraph_vector_clear(evec);

			if (path_costs) {
				VECTOR(*path_costs)[i] = u_data->distance - 1.0;
			}

			size = u_data->length;
			igraph_vector_resize(evec, size);

			act = u;
			while ((data + act)->parent) {
				edge = (data + act)->parent - 1;
				act = IGRAPH_OTHER(graph, edge, act);
				size--;
				VECTOR(*evec)[size] = edge;
			}
		}
	}

	/* de-allocate */
	igraph_vit_destroy(&tovit);
	igraph_dqueue_destroy(&Q);
	igraph_dqueue_destroy(&Q_pape);
	igraph_lazy_inclist_destroy(&inclist);
	igraph_Free(data);

	return 0;
}
