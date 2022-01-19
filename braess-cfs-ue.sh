set -x
readonly TIME_LIMIT_H=2.0
readonly SOLVER=GUROBI
readonly EPS=1e-3
readonly BETA=1e12
readonly ARGS="--result-kind fw --mip --no-cfs-lp --no-mcr --no-cfs-so --epsilon-user $EPS --epsilon-fleet $EPS --beta $BETA --use-all-paths --solver $SOLVER --time-limit $TIME_LIMIT_H --verbose"
readonly RESULTS="examples/results/"
readonly CFS="python examples/run-critical-fleet-size.py"


echo "SCALE 1.0"
echo "========="
$CFS $ARGS -o braess-cfs-ue $RESULTS/braess-ue/results-2021-03-05T13:45:58/
echo ""
