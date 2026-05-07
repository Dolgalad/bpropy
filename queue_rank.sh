#!/usr/bin/env bash
set -euo pipefail

PARTITION="${1:-gpu_p13}"
USER_NAME="${USER}"

# Collect pending jobs in partition, with their sprio priority (numeric)
# sprio output example columns: JOBID PARTITION PRIORITY ...
# We parse: jobid, partition, priority
mapfile -t ALL < <(sprio -l 2>/dev/null | awk -v p="$PARTITION" '
  NR==1 {next}
    $2==p && $1 ~ /^[0-9]+/ {print $1, $2, $3}
    ')

    if [[ ${#ALL[@]} -eq 0 ]]; then
	      echo "No pending jobs found in partition '$PARTITION' (or sprio unavailable)."
	        exit 0
    fi

    # Sort by priority descending, then jobid ascending for stability
    SORTED=$(printf "%s\n" "${ALL[@]}" | sort -k3,3nr -k1,1n)

    # Build rank map: jobid -> rank
    declare -A RANK
    rank=0
    while read -r jid part pr; do
	      rank=$((rank+1))
	        RANK["$jid"]="$rank"
done <<< "$SORTED"

# Get your jobs (including arrays): squeue gives array ids like 413608_[0-2]
# We'll expand to base jobid (413608) for matching sprio jobids,
# but we’ll also try to match full ids if sprio includes array elements on your system.
mapfile -t MYJOBS < <(squeue -u "$USER_NAME" -p "$PARTITION" -t PD -h -o "%i %j %R")

if [[ ${#MYJOBS[@]} -eq 0 ]]; then
  echo "You have no PENDING jobs in partition '$PARTITION'."
  exit 0
fi

echo "Partition: $PARTITION"
echo "User:      $USER_NAME"
echo
printf "%-18s %-6s %-10s %-24s %s\n" "JOBID" "RANK" "PRIORITY" "NAME" "REASON"
echo "------------------------------------------------------------------------------------------"

while read -r jid name reason; do
  # Try direct match
  r="${RANK[$jid]:-}"

  # If not found, try stripping array suffix like _[0-2] or _0
  if [[ -z "${r}" ]]; then
      base="${jid%%_*}"
      r="${RANK[$base]:-}"
  fi

  # Get priority for this job from sprio (best-effort)
  pr=$(printf "%s\n" "$SORTED" | awk -v j="$jid" -v b="${jid%%_*}" '
        $1==j {print $3; found=1}
        $1==b && !found {print $3; found=1}
       ' | head -n1)

  printf "%-18s %-6s %-10s %-24s %s\n" "$jid" "${r:-?}" "${pr:-?}" "$name" "$reason"
        done < <(printf "%s\n" "${MYJOBS[@]}")
  echo
  echo "Rank = position among all jobs in '$PARTITION' ordered by sprio priority (1 = highest)."
