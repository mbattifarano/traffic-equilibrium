set -x
readonly TIME_LIMIT_H=2.0
readonly SOLVER=GUROBI
readonly EPS=1e-8
readonly BETA=1.0
readonly ARGS="--result-kind fw --no-abandon --no-mip --no-cfs-lp --no-mcr --cfs-ue-lp --link-error-as-constraint --epsilon-fleet-marginal-cost 1e-3 --epsilon-user $EPS --epsilon-fleet $EPS --beta $BETA --solver $SOLVER --verbose --time-limit $TIME_LIMIT_H --mip-solutions 1"
readonly RESULTS="examples/results/"
readonly CFS="python examples/run-critical-fleet-size.py"

echo "SCALE 1.0 (PGH)"
echo "==============="
$CFS $ARGS -o spc-greater-pgh-1.0-cfs-ue-eps-$EPS-beta-$BETA-final examples/results/pittsburgh-network-mpo-greater-pgh-1.0-min-50.0-NoThruTrips_fw-ue/results-2021-10-20T21:09:25