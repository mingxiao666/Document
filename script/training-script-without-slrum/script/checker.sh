#!/bin/bash

set_cpu_freq_userspace(){
  for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo userspace > $i; done
  for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do echo 3330000 > $i; done
  for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do echo 3000000 > $i; done
  for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed; do echo 3200000 > $i; done
  echo 0 > /sys/devices/system/cpu/cpufreq/boost
  ret=($(cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_cur_freq|sort -n)); echo CPUFREQ: "MIN=$[${ret[0]}/1000] MAX=$[${ret[-1]}/1000]"
}

global_status=0

## CPU
if [ $(grep processor /proc/cpuinfo|wc -l) -ne 144 ]; then
  echo 'ERROR: CPU CORES is not 144'
  global_status=$[global_status+1]
fi
cpupower -c all frequency-set -g performance &>/dev/null
if [ $(grep -v performance /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor|wc -l) -ne 0 ]; then
  echo 'ERROR: CPU governor is not performance'
  global_status=$[global_status+1]
fi
s0=$(cat /sys/devices/system/cpu/cpu{0..71}/cpufreq/cpuinfo_cur_freq|awk 'BEGIN{sum=0}{sum+=$1}END{print int(sum/NR/1000)}')
s1=$(cat /sys/devices/system/cpu/cpu{72..143}/cpufreq/cpuinfo_cur_freq|awk 'BEGIN{sum=0}{sum+=$1}END{print int(sum/NR/1000)}')
if [ ${s0} -lt 500 ]; then
  echo 'ERROR: CPU socket0 average frequency is abnormal(<500MHz)'
  global_status=$[global_status+1]
fi
if [ ${s1} -lt 500 ]; then
  echo 'ERROR: CPU socket1 average frequency is abnormal(<500MHz)'
  global_status=$[global_status+1]
fi

## MEM
if [ $(dmidecode -t memory|grep -P '\tSize:'|tr -d '\t'|grep 480|wc -l) -ne 2 ]; then
  echo 'ERROR: Memory size is not 2*480GB'
  global_status=$[global_status+1]
fi

## GPU
if [ $(lspci | grep '3D controller' | wc -l) -ne 4 ]; then
  echo 'ERROR: GPU number is not 4'
  global_status=$[global_status+1]
fi
if [ $(lspci | grep '3D controller' | grep 'rev ff' | wc -l) -gt 0 ]; then
  echo 'ERROR: GPU met ref ff issue'
  global_status=$[global_status+1]
fi
if `nvidia-smi -L` &>/dev/null; then
  echo "ERROR: GPU driver didn't load correctly"
  global_status=$[global_status+1]
fi
if [ $(nvidia-smi nvlink -s|grep '5[0-9].* GB/s'|wc -l) -ne 72 ]; then
  echo 'ERROR: 72 GPU NVLINKs are not fully connected'
  global_status=$[global_status+1]
fi
if [ $(nvidia-smi --format=csv --query-gpu gpu_bus_id,fabric.status|grep 'GPU requires reset'|wc -l
) -ne 0 ]; then
  echo 'ERROR: GPU needs reset to recover the NVLINKs'
  global_status=$[global_status+1]
fi

## IMEX
if [ $(ls -1 /dev/nvidia-caps-imex-channels/|wc -l) -lt 1 ]; then
  echo 'ERROR: IMEX channel doesnt exist'
  global_status=$[global_status+1]
fi
if [ ! -f /etc/nvidia-imex/nodes_config.cfg ]; then
  echo 'ERROR: IMEX /etc/nvidia-imex/nodes_config.cfg doesnt exist. Restart nvidia-imex-config and nvidia-imex service'
  global_status=$[global_status+1]
fi
if [ $(nvidia-imex-ctl -q 2>/dev/null) != "READY" ]; then
  echo 'ERROR: IMEX is not active'
  global_status=$[global_status+1]
fi

## IB
#if [ $(lspci | grep 'Infiniband controller'|wc -l) -ne 4 ]; then
#  echo 'ERROR: IB HCA number is not 4'
#  global_status=$[global_status+1]
#fi
if [ $(lspci | grep 'Ethernet.*BlueField-3'|wc -l) -ne 4 ]; then
  echo 'ERROR: BlueField-3 ethernet ports number is not 4'
  global_status=$[global_status+1]
fi
#active_hcas=($(lspci -D|grep 'Infiniband controller'|awk '{print $1}'|while read i; do hca=$(basename $(ls -l /sys/class/infiniband|grep -o ${i}.*)); grep ACTIVE /sys/class/infiniband/${hca}/ports/1/state &>/dev/null && echo ${hca}:1;done))
#if [ ${#active_hcas[*]} -ne 4 ]; then
#  echo "WARNING: Active HCA number is not 4" # $(echo ${active_hcas[*]}|tr ' ' ',')
#fi
#hcas=($(lspci -D|grep 'Infiniband controller'|awk '{print $1}'|while read i; do echo $(basename $(ls -l /sys/class/infiniband|grep -o ${i}.*));done))
#ret=($(
#for dev in ${hcas[*]}; do
#  link_downed=$(</sys/class/infiniband/${dev}/ports/1/counters/link_downed)
#  symbol_error=$(</sys/class/infiniband/${dev}/ports/1/counters/symbol_error)
#  if [[ ${link_downed} -ne 0 || ${symbol_error} -ne 0 ]]; then
#    echo ${dev}:link_downed=${link_downed},symbol_error=${symbol_error}
#  fi
#done
#))
#if [ ${#ret[*]} -ne 0 ]; then
#  echo "WARNING: ${ret[*]}"
#fi

## NFS
if `df -Th|grep master:/home` &>/dev/null; then
  echo 'ERROR: home not mounted'
  global_status=$[global_status+1]
fi
if `df -Th|grep master:/cm/shared` &>/dev/null; then
  echo 'ERROR: cmshared not mounted'
  global_status=$[global_status+1]
fi

#CUDA Test
if [ $(nvidia-smi topo -p2p n|grep GPU[0-9].*OK|wc -l) -eq 0 ]; then
    echo 'ERROR: GPU P2P is disabled'
    global_status=$[global_status+1]
fi
if [ -x /home/cmsupport/workspace/cudatest/foo ]; then
  timeout 3 /home/cmsupport/workspace/cudatest/foo &>/dev/null
  ret=$?
  if [ $ret -ne 0 ]; then
    echo 'ERROR: Cannot run even a simple CUDA app'
    global_status=$[global_status+1]
  fi
fi

## Print health check status
if [ ${global_status} -ne 0 ]; then
  echo 'INFO: Health check FAILED'
else
  echo 'INFO: Health check PASSED'
fi

##
exit 0
