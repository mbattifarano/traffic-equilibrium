typedef struct node_data {
	long int parent;    // previous node in the shortest path
	double distance;    // distance of the shortest path to this node
	int enqueued;		// 1 if this node is currently in the queue
	int visited;        // 1 if this node has been queued before
} node_data_t;

static int dqueue_push_front(igraph_dqueue_t *q, igraph_real_t elem) {
	assert(q != 0);
	assert(q->stor_begin != 0);
	assert(q->begin != q->end); // assert not full
	if (q->end == NULL) {
		// the dqueue is empty just do a normal push
		q->end = q->begin;
		*(q->end) = elem;
		(q->end)++;
		if (q->end == q->stor_end) {
			q->end = q->stor_begin;
		}
	} else {
		// the dqueue has elements
		if (q->begin == q->stor_begin) {
			// the front of the queue is at the beginning of storage
			q->begin = q->stor_end;
		}
		(q->begin)--;
		*(q->begin) = elem;
	}
	return 0;
}

static int get_shortest_paths_bellman_ford(
	const igraph_t *graph,
	dang_t* path_trie,
	igraph_real_t *path_costs, // cost of each path (optional)
	const long int source,
	igraph_vs_t to,
	const igraph_vector_t *weights,
	igraph_real_t *link_flow,
	igraph_real_t *volumes
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
	long int i, k, nlen, nei, size, act, edge, head;

	igraph_dqueue_t Q1, Q2;
	igraph_dqueue_t *Q;
	long int q2_count = 1, q1_count = 0;
	igraph_vit_t tovit;
	igraph_vector_t path;
	igraph_vector_init(&path, igraph_ecount(graph));

	igraph_vector_t *neis;
	node_data_t *data = igraph_Calloc(no_of_nodes, node_data_t);

	long int u, v;
	int empty;
	node_data_t *u_data, *v_data;
	igraph_real_t dist_to_v, tmp_dist;

	dang_node_t **walk;
	dang_node_t **prev;

	igraph_dqueue_init(&Q1, no_of_nodes);
	igraph_dqueue_init(&Q2, no_of_nodes);
	igraph_lazy_inclist_init(graph, &inclist, IGRAPH_OUT);
	igraph_vit_create(graph, to, &tovit);

	/* bellman-ford */
	igraph_dqueue_push(&Q2, source);
	u_data = (data + source);
	u_data->enqueued = 1;
	u_data->visited = 1;
	u_data->distance = 1.0;

	/* Run shortest paths */
	while (q2_count + q1_count) {
		if (q1_count) {
			Q = &Q1;
			q1_count--;
		} else {
			Q = &Q2;
			q2_count--;
		}
		u = (long int) igraph_dqueue_pop(Q);
		u_data = (data + u);

		u_data->enqueued--;

		neis = igraph_lazy_inclist_get(&inclist, (igraph_integer_t) u);
		nlen = igraph_vector_size(neis);

		for (k = 0; k < nlen; k++) {
			nei = (long int) VECTOR(*neis)[k];
			v = IGRAPH_OTHER(graph, nei, u);
			v_data = (data + v);
			dist_to_v = v_data->distance;
			tmp_dist = u_data->distance + VECTOR(*weights)[nei];
			if ((tmp_dist < dist_to_v) || (dist_to_v == 0.0) ) {
				/* relax the edge */
				v_data->distance = tmp_dist;
				v_data->parent = nei + 1;

				if (!(v_data->enqueued)) {
					v_data->enqueued++;

					/* Pape's rule + SLF
					 *
					 * If a node has been visited previously, add it to Q1
					 * else, add it to Q2.
					 * If the label is smaller than the head label push it to
					 * the front of the queue, else, push to the back
					 */
					if (v_data->visited) {
						Q = &Q1;
						empty = (q1_count == 0);
						q1_count++;
					} else {
						Q = &Q2;
						empty = (q2_count == 0);
						q2_count++;
						v_data->visited = 1;
					}

					if (empty) {
						igraph_dqueue_push(Q, v);
					} else {
						head = (long int) igraph_dqueue_head(Q);
						if ( tmp_dist < ((data + head)->distance) ) {
							dqueue_push_front(Q, v);
						} else {
							igraph_dqueue_push(Q, v);
						}
					}
				}
			}
		}
	}

	/* populate edges and path costs */
	if (path_trie) {
		for (IGRAPH_VIT_RESET(tovit), i = 0; !IGRAPH_VIT_END(tovit); IGRAPH_VIT_NEXT(tovit), i++) {
			u = IGRAPH_VIT_GET(tovit);
			u_data = (data + u);

			//igraph_vector_t *evec = 0;
			//evec = VECTOR(*edges)[i];
			//igraph_vector_clear(evec);

			if (path_costs) {
				path_costs[i] = u_data->distance - 1.0;
			}

			igraph_vector_resize(&path, 0);
			act = u;
			while ((data + act)->parent) {
				edge = (data + act)->parent - 1;
				// push edge onto stack
				igraph_vector_push_back(&path, edge);
				// Add volume to link flow vector
				link_flow[edge] += volumes[i];
				act = IGRAPH_OTHER(graph, edge, act);
			}
			size = igraph_vector_size(&path);

			k = 0;
			walk = &(path_trie->root);
			prev = walk;
			while (k < size) {
				prev = walk;
				edge = VECTOR(path)[size - k - 1];
				walk = dang_append_to_path(path_trie, walk, edge, &k);
			}
			// store target node id as value
			dang_mark_end(path_trie, prev, u);
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
