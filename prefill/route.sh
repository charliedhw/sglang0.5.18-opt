python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://${P_HOST:?export P_HOST=<prefill node IP>}:30001 \
  --decode http://${D_HOST:?export D_HOST=<decode node IP>}:30002 \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30000 \
  --request-timeout-secs 72000 \
  2>&1 | tee log_route
