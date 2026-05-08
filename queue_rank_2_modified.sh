#!/usr/bin/env bash
set -euo pipefail

PARTITION="${1:-gpu_p13}"
USER_NAME="${USER}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

sprio_file="$tmpdir/sprio.tsv"
queue_file="$tmpdir/queue.tsv"

# Pull exact priority components in a parseable format.
# Fields:
# jobid|partition|priority|age|fairshare|jobsize|partition_factor|qos
sprio -p "$PARTITION" -o "%i|%r|%Y|%A|%F|%J|%P|%Q" 2>/dev/null \
  | awk 'NR>1 && $1 ~ /^[0-9]+/ {print}' \
  > "$sprio_file"

if [[ ! -s "$sprio_file" ]]; then
  echo "No pending jobs found in partition '$PARTITION' (or sprio unavailable)."
  exit 0
fi

# Get all of *your* pending jobs with useful scheduling hints.
# %i jobid
# %j name
# %Q priority long
# %S expected start time
# %R reason list
# %D nodes
# %C cpus
# %L time left / for PD often time limit-ish display may vary by site
# %q qos
squeue -u "$USER_NAME" -p "$PARTITION" -t PD -h \
  -o "%i|%j|%Q|%S|%R|%D|%C|%q" \
  > "$queue_file"

if [[ ! -s "$queue_file" ]]; then
  echo "You have no PENDING jobs in partition '$PARTITION'."
  exit 0
fi

echo "Partition: $PARTITION"
echo "User:      $USER_NAME"
echo

awk -F'|' '
  FNR==NR {
    jid=$1
    part=$2
    prio=$3
    age=$4
    fair=$5
    jsize=$6
    pfac=$7
    qosf=$8

    data[jid]=prio "|" age "|" fair "|" jsize "|" pfac "|" qosf
    order[++n]=jid
    next
  }
  {
    qjid=$1
    name=$2
    qprio=$3
    start=$4
    reason=$5
    nodes=$6
    cpus=$7
    qos=$8

    base=qjid
    sub(/_.*/, "", base)

    pr=""
    age=""
    fair=""
    jsize=""
    pfac=""
    qosf=""

    if (qjid in data) {
      split(data[qjid], a, "|")
    } else if (base in data) {
      split(data[base], a, "|")
    } else {
      split("", a)
    }

    pr   = (a[1] != "" ? a[1] : qprio)
    age  = a[2]
    fair = a[3]
    jsize= a[4]
    pfac = a[5]
    qosf = a[6]

    rank="?"
    for (i=1; i<=n; i++) {
      if (order[i] == qjid || order[i] == base) {
        rank=i
        break
      }
    }

    row_count++
    rows[row_count]=sprintf("%-18s %-6s %-12s %-8s %-10s %-8s %-8s %-8s %-20s %-20s %s",
      qjid, rank, pr, age, fair, jsize, pfac, qosf, start, reason, name)

    if (rank != "?" && (best_rank == "" || rank + 0 < best_rank + 0)) {
      best_rank=rank
      best_jobid=qjid
      best_name=name
      best_start=start
    }
  }
  END {
    if (best_rank != "") {
      print "Next likely job to start:"
      printf "  Name:       %s\n", best_name
      printf "  Start time: %s\n", best_start
      printf "  Rank:       %s\n", best_rank
      printf "  Job ID:     %s\n", best_jobid
    } else {
      print "Next likely job to start: unavailable; no matching rank found in sprio output."
    }

    print ""
    printf "%-18s %-6s %-12s %-8s %-10s %-8s %-8s %-8s %-20s %-20s %s\n", \
      "JOBID", "RANK", "PRIORITY", "AGE", "FAIRSHARE", "JSIZE", "PART", "QOS", "START_TIME", "RES", "NAME"
    print "--------------------------------------------------------------------------------------------------------------------------------------------"

    for (i=1; i<=row_count; i++) {
      print rows[i]
    }
  }
' "$sprio_file" "$queue_file"

echo
echo "Notes:"
echo "  - The top summary shows your pending job with the smallest numeric RANK."
echo "  - RANK is among pending jobs in partition '$PARTITION' sorted by Slurm priority."
echo "  - START_TIME is Slurm's expected start time if the scheduler can estimate one."
echo "  - FAIRSHARE is often the biggest reason similarly-aged jobs move around."
echo "  - REASON=Priority means higher-priority jobs currently block this one."
