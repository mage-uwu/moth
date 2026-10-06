# Moved to vision/job.sh; kept so a pod started with "bash pod/vl_job.sh" still runs it after a restart.
exec bash vision/job.sh "$@"
