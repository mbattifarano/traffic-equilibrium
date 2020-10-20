typedef struct node_data {
	long int parent;    // previous node in the shortest path
	double distance;    // distance of the shortest path to this node
	long int length;    // number of edges in the shortest paths to this node
	int visited;        // 1 if this node has been queued before
	int dirty;          // 1 if this node was relabeled on the last iteration
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

	igraph_dqueue_t Q1, Q2;
	igraph_dqueue_t *Q;
	long int q2_count = 1, q1_count = 0;
	igraph_vit_t tovit;

	igraph_vector_t *neis;
	node_data_t *data = igraph_Calloc(no_of_nodes, node_data_t);

	long int u, v;
	node_data_t *u_data, *v_data;
	igraph_real_t dist_to_v;
	igraph_real_t tmp_dist;

	igraph_dqueue_init(&Q1, no_of_nodes);
	igraph_dqueue_init(&Q2, no_of_nodes);
	igraph_lazy_inclist_init(graph, &inclist, mode);
	igraph_vit_create(graph, to, &tovit);

	/* bellman-ford */
	igraph_dqueue_push(&Q2, source);
	u_data = (data + source);
	u_data->visited = 1;
	u_data->distance = 1.0;

	/* Run shortest paths */
	while (q2_count + q1_count) {
		// pop from pape's queue if non-empty
		if (q1_count) {
			Q = &Q1;
			q1_count--;
		} else { // pop from the regular queue
			Q = &Q2;
			q2_count--;
		}
		u = (long int) igraph_dqueue_pop(Q);
		u_data = (data + u);

		u_data->dirty = 0; // set u to clean
		// dist_to_u = u_data->distance; // 0 means unreachable

		neis = igraph_lazy_inclist_get(&inclist, (igraph_integer_t) u);
		nlen = igraph_vector_size(neis);

		for (k = 0; k < nlen; k++) {
			nei = (long int) VECTOR(*neis)[k];
			v = IGRAPH_OTHER(graph, nei, u);
			v_data = (data + v);
			dist_to_v = v_data->distance;
			tmp_dist = u_data->distance + VECTOR(*weights)[nei];
			if ((tmp_dist < dist_to_v) || (!dist_to_v) ) {
				/* relax the edge */
				v_data->distance = tmp_dist;
				v_data->parent = nei + 1;
				v_data->length = u_data->length + 1;

				if (!(v_data->dirty)) {
					v_data->dirty = 1;
					/* Pape's rule:
					* If the target has been scanned already,
					* add it to pape's queue to ensure it is drawn first
					* otherwise add it to the regular queue.
					*/
					if (v_data->visited) {
						Q = &Q1;
						q1_count++;
					} else {
						Q = &Q2;
						q2_count++;
						v_data->visited = 1;
					}
					igraph_dqueue_push(Q, v);
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
	igraph_dqueue_destroy(&Q1);
	igraph_dqueue_destroy(&Q2);
	igraph_lazy_inclist_destroy(&inclist);
	igraph_Free(data);

	return 0;
}
