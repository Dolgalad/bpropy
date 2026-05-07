#!/usr/bin/env bash
set -u

INTERVAL="${1:-5}"

build_log_path() {
  local jobdesc="$1"

  local jobid jobname workdir array_jobid array_taskid
  jobid="$(grep -o 'JobId=[^ ]*' <<< "$jobdesc" | head -n1 | cut -d= -f2)"
  jobname="$(grep -o 'JobName=[^ ]*' <<< "$jobdesc" | head -n1 | cut -d= -f2)"
  workdir="$(grep -o 'WorkDir=[^ ]*' <<< "$jobdesc" | head -n1 | cut -d= -f2)"
  array_jobid="$(grep -o 'ArrayJobId=[^ ]*' <<< "$jobdesc" | head -n1 | cut -d= -f2 || true)"
  array_taskid="$(grep -o 'ArrayTaskId=[^ ]*' <<< "$jobdesc" | head -n1 | cut -d= -f2 || true)"

  if [ -z "$jobname" ] || [ -z "$workdir" ]; then
    return 1
  fi

  local base_dir
  base_dir="$(dirname "$workdir")/slurm_outputs/$jobname"

  if [ -n "${array_jobid:-}" ] && [ -n "${array_taskid:-}" ]; then
    echo "$base_dir/${jobname}_${array_jobid}_${array_taskid}.out"
  else
    echo "$base_dir/${jobname}_${jobid}.out"
  fi
}

while true; do
  clear
  date
  echo
  echo "=== squeue -u $USER ==="
  squeue -u "$USER"
  echo

  mapfile -t job_ids < <(squeue -h -u "$USER" -t RUNNING -o "%i")

  if [ "${#job_ids[@]}" -eq 0 ]; then
    echo "No running jobs."
    sleep "$INTERVAL"
    continue
  fi

  for jobid in "${job_ids[@]}"; do
    jobdesc="$(scontrol show job "$jobid" 2>/dev/null || true)"
    [ -n "$jobdesc" ] || continue

    jobname="$(grep -o 'JobName=[^ ]*' <<< "$jobdesc" | head -n1 | cut -d= -f2)"
    logfile="$(build_log_path "$jobdesc" || true)"

    echo "============================================================"
    echo "Job: ${jobid}"
    echo "Name: ${jobname:-unknown}"
    echo "Log:  ${logfile:-not found}"
    echo "--- tail -n 1 ---"

    if [ -n "${logfile:-}" ] && [ -f "$logfile" ]; then
      tail -n 1 "$logfile"
    else
      echo "No readable log file found."
    fi
    echo
  done

  sleep "$INTERVAL"
done
