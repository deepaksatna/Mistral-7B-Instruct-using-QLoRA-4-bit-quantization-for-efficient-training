#!/bin/bash
# Monitor GPU utilization across all 4 benchmark nodes
# Displays real-time GPU stats in a compact format

PODS=("benchmark-node-1" "benchmark-node-2" "benchmark-node-3" "benchmark-node-4")

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to get GPU stats from a pod
get_gpu_stats() {
    local pod=$1

    kubectl exec $pod -- nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "N/A,N/A,N/A,N/A,N/A,N/A"
}

# Function to colorize GPU utilization
colorize_util() {
    local util=$1

    if [ "$util" == "N/A" ]; then
        echo -e "${RED}N/A${NC}"
    elif [ "$util" -ge 80 ]; then
        echo -e "${GREEN}${util}%${NC}"
    elif [ "$util" -ge 50 ]; then
        echo -e "${YELLOW}${util}%${NC}"
    else
        echo -e "${RED}${util}%${NC}"
    fi
}

# Clear screen and show header
clear
echo "======================================================================="
echo "GPU Monitoring - 4× A10 GPU Nodes"
echo "======================================================================="
echo "Press Ctrl+C to stop"
echo ""

# Main monitoring loop
while true; do
    echo -e "\n$(date '+%Y-%m-%d %H:%M:%S')"
    echo "-----------------------------------------------------------------------"
    printf "%-18s %-10s %-10s %-15s %-8s %-10s\n" "NODE" "GPU %" "MEM %" "MEMORY" "TEMP" "POWER"
    echo "-----------------------------------------------------------------------"

    for i in {1..4}; do
        pod="benchmark-node-$i"

        # Get stats
        stats=$(get_gpu_stats $pod)

        # Parse stats
        IFS=',' read -r gpu_util mem_util mem_used mem_total temp power <<< "$stats"

        # Format output
        if [ "$gpu_util" != "N/A" ]; then
            gpu_util_colored=$(colorize_util $gpu_util)
            mem_util_colored=$(colorize_util $mem_util)
            mem_display="${mem_used}/${mem_total} MB"
            temp_display="${temp}°C"
            power_display="${power} W"
        else
            gpu_util_colored="${RED}N/A${NC}"
            mem_util_colored="${RED}N/A${NC}"
            mem_display="N/A"
            temp_display="N/A"
            power_display="N/A"
        fi

        printf "%-18s %-20s %-20s %-15s %-8s %-10s\n" \
            "$pod" \
            "$gpu_util_colored" \
            "$mem_util_colored" \
            "$mem_display" \
            "$temp_display" \
            "$power_display"
    done

    echo "-----------------------------------------------------------------------"

    # Show benchmark process status
    echo ""
    echo "Benchmark Status:"
    for i in {1..4}; do
        pod="benchmark-node-$i"

        # Check if benchmark is running
        pid_check=$(kubectl exec $pod -- bash -c "
            if [ -f /tmp/benchmark_pid_qlora ]; then
                pid=\$(cat /tmp/benchmark_pid_qlora)
                if ps -p \$pid > /dev/null 2>&1; then
                    echo 'RUNNING'
                else
                    echo 'FINISHED'
                fi
            else
                echo 'NOT STARTED'
            fi
        " 2>/dev/null || echo "N/A")

        case $pid_check in
            RUNNING)
                echo -e "  Node $i: ${GREEN}● Running${NC}"
                ;;
            FINISHED)
                echo -e "  Node $i: ${BLUE}✓ Finished${NC}"
                ;;
            "NOT STARTED")
                echo -e "  Node $i: ${YELLOW}○ Not started${NC}"
                ;;
            *)
                echo -e "  Node $i: ${RED}✗ Unknown${NC}"
                ;;
        esac
    done

    # Wait before next update
    sleep 5

    # Move cursor up to overwrite (comment out for log-style output)
    # tput cuu 15
done
