# GLUE (pod/glue_job.sh), then the typed-decisions probe (pod/decide_job.sh), on one GPU pod.
bash pod/glue_job.sh
bash pod/decide_job.sh
echo "=== ALL DONE $(date -u +%T) ==="
