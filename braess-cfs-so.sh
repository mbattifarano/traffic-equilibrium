set -x
readonly TIME_LIMIT_H=2.0
readonly SOLVER=GUROBI
readonly EPS=1e-3
readonly BETA=1e12
readonly ARGS="--result-kind fw --mip --cfs-lp --mcr --no-cfs-ue-lp --epsilon-user $EPS --epsilon-fleet $EPS --beta $BETA --use-all-paths --solver $SOLVER --time-limit $TIME_LIMIT_H --verbose"
readonly RESULTS="examples/results/"
readonly CFS="python examples/run-critical-fleet-size.py"


$CFS $ARGS -o braess-cfs-so $RESULTS/braess-so/results-2021-12-15T20:55:35
