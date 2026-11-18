from pathlib import Path
#Path is Python's modern way to handle file/folder locations that works the same on Windows, Mac, and Linux (instead of manually typing \ or /).
#python
import numpy as np
import pandas as pd

from utils import bfs, dfs, ucs, plot_route_paths, load_openflights, haversine_km, route_is_valid


ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "all_datasets"
RESULTS = ROOT / "results" / "lab01"
RESULTS.mkdir(parents=True, exist_ok=True)


# Q1

airports, routes, coords, graph, stats = load_openflights(
    DATA / "lab01-03_airports.csv", DATA / "lab01-03_routes.csv"
)

queries = pd.read_csv("utils/search_queries.csv").head(3)

print(pd.Series(stats, name="count").to_frame())
print(queries)

# Q2

sample = pd.DataFrame(graph["KUL"][:5], columns=["destination", "distance_km"])

direct_km = haversine_km(
    coords["KUL"]["latitude"], coords["KUL"]["longitude"],
    coords["KEF"]["latitude"], coords["KEF"]["longitude"],
)

# Q3

rows = []
for query in queries.itertuples():
    for name, algorithm in [("BFS", bfs), ("DFS", dfs), ("UCS", ucs)]:
        result = algorithm(graph, query.start, query.goal)
    rows.append({
            "query_id": query.query_id, "start": query.start, "goal": query.goal,
            "algorithm": name, **result.as_dict(), "path": " -> ".join(result.path),
        })

metrics = pd.DataFrame(rows).drop(columns="found")
metrics.to_csv(RESULTS / "metrics.csv", index=False)

# Q4

edge_km = {airport: dict(neighbours) for airport, neighbours in graph.items()}

for row in metrics.itertuples():
    path = row.path.split(" -> ")
    assert route_is_valid(path, graph), f"{row.algorithm} {row.query_id}: a leg is not a real route"
    recomputed = sum(edge_km[a][b] for a, b in zip(path, path[1:]))
    assert np.isclose(recomputed, row.path_cost_km), f"{row.algorithm} {row.query_id}: cost mismatch"


# Q5

print(metrics.pivot(index="query_id", columns="algorithm", values=["hops", "path_cost_km"]).round(0))

reversed_graph = {airport: neighbours[::-1] for airport, neighbours in graph.items()}

for query in queries.itertuples():
    original = dfs(graph, query.start, query.goal).hops
    reordered = dfs(reversed_graph, query.start, query.goal).hops
    print(f"{query.query_id} DFS: {original} hops with the original order, "
          f"{reordered} with the reversed order")


# Q6

q1 = metrics[metrics.query_id == "Q1"]

paths = {row.algorithm: row.path.split(" -> ") for row in q1.itertuples()}

figure, _ = plot_route_paths(paths, coords, title="Route comparison: KUL to KEF")
figure.savefig(RESULTS / "route_comparison.png", dpi=160, bbox_inches="tight")

