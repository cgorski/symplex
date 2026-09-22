//! Directed-graph primitives on adjacency lists over `0..n` (std only).
//!
//! The Markov-chain code asks for communication classes, the rating
//! aggregators for strong connectivity; both are strongly connected
//! components of a small directed graph, computed here once, iteratively,
//! in linear time.

/// The strongly connected components of the directed graph whose vertex
/// `v` has the out-neighbours `adj[v]` (an out-of-range neighbour is
/// ignored; duplicates are harmless).
///
/// Iterative Tarjan in `O(V + E)`.  The result is canonical: every
/// component lists its vertices in ascending order and the components are
/// ordered by their smallest vertex — the order a scan of `0..n` that
/// groups each vertex with those reaching it and reached by it would
/// produce.  For `0 ⇄ 1 → 2 ⇄ 3` with `4` isolated the components are
/// `[[0, 1], [2, 3], [4]]`.
pub(crate) fn strongly_connected_components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    const UNVISITED: usize = usize::MAX;
    let mut index = vec![UNVISITED; n];
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut next_index = 0usize;
    // Explicit DFS frames: (vertex, position in its adjacency list).
    let mut frames: Vec<(usize, usize)> = Vec::new();

    for root in 0..n {
        if index[root] != UNVISITED {
            continue;
        }
        index[root] = next_index;
        lowlink[root] = next_index;
        next_index += 1;
        stack.push(root);
        on_stack[root] = true;
        frames.push((root, 0));

        while let Some(&mut (v, ref mut pos)) = frames.last_mut() {
            if let Some(&w) = adj[v].get(*pos) {
                *pos += 1;
                if w >= n {
                    continue;
                }
                if index[w] == UNVISITED {
                    index[w] = next_index;
                    lowlink[w] = next_index;
                    next_index += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    frames.push((w, 0));
                } else if on_stack[w] {
                    lowlink[v] = lowlink[v].min(index[w]);
                }
                continue;
            }
            // Every edge of `v` is explored.
            frames.pop();
            if lowlink[v] == index[v] {
                let mut component = Vec::new();
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    component.push(w);
                    if w == v {
                        break;
                    }
                }
                component.sort_unstable();
                components.push(component);
            }
            if let Some(&(parent, _)) = frames.last() {
                lowlink[parent] = lowlink[parent].min(lowlink[v]);
            }
        }
    }

    components.sort_unstable_by_key(|c| c.first().copied().unwrap_or(usize::MAX));
    components
}

#[cfg(test)]
mod tests {
    use super::strongly_connected_components;

    /// Reference: Warshall closure, then group `i` with every `j` that
    /// reaches and is reached by it.
    fn by_closure(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
        let n = adj.len();
        let mut reach = vec![vec![false; n]; n];
        for (i, out) in adj.iter().enumerate() {
            for &j in out {
                reach[i][j] = true;
            }
        }
        for k in 0..n {
            let via_k = reach[k].clone();
            for row in reach.iter_mut() {
                if !row[k] {
                    continue;
                }
                for (cell, &through) in row.iter_mut().zip(&via_k) {
                    if through {
                        *cell = true;
                    }
                }
            }
        }
        let mut assigned = vec![false; n];
        let mut classes = Vec::new();
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let class: Vec<usize> = (i..n)
                .filter(|&j| j == i || (reach[i][j] && reach[j][i]))
                .collect();
            for &j in &class {
                assigned[j] = true;
            }
            classes.push(class);
        }
        classes
    }

    #[test]
    fn documented_example() {
        let adj = vec![vec![1], vec![0, 2], vec![3], vec![2], vec![]];
        assert_eq!(
            strongly_connected_components(&adj),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
    }

    #[test]
    fn empty_and_singletons() {
        assert_eq!(strongly_connected_components(&[]), Vec::<Vec<usize>>::new());
        assert_eq!(strongly_connected_components(&[vec![]]), vec![vec![0]]);
        assert_eq!(strongly_connected_components(&[vec![0]]), vec![vec![0]]);
    }

    #[test]
    fn matches_closure_on_random_graphs() {
        // Deterministic LCG so the test is reproducible.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for _ in 0..200 {
            let n = 1 + next() % 9;
            let density = 1 + next() % 4;
            let adj: Vec<Vec<usize>> = (0..n)
                .map(|_| (0..n).filter(|_| next() % 4 < density).collect())
                .collect();
            assert_eq!(
                strongly_connected_components(&adj),
                by_closure(&adj),
                "adj = {adj:?}"
            );
        }
    }

    #[test]
    fn reversed_labels_still_ascending() {
        // 3 → 2 → 1 → 0 → 3: one component, listed ascending.
        let adj = vec![vec![3], vec![0], vec![1], vec![2]];
        assert_eq!(strongly_connected_components(&adj), vec![vec![0, 1, 2, 3]]);
    }
}
