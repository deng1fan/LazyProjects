from general_files.utils.common_util import RedisClient
import psutil
import json
import time
from nvitop import Device, GpuProcess, NA
import setproctitle

setproctitle.setproctitle("Redis Watcher!!!")


redis_client = RedisClient()
while True:
    self_occupied_gpus = redis_client.get_self_occupied_gpus(only_gpus=False)
    queue = redis_client.client.lrange('wait_queue', 0, -1)
    for task in self_occupied_gpus:
        pid = int(task['system_pid'])
        if not psutil.pid_exists(pid):
            redis_client.client.hdel("self_occupied_gpus", task['task_id'])
            print(f"发现 GPU占用信息 中存在残余数据，已清除，进程为{pid}")
    for task_json in queue:
        task = json.loads(task_json)
        pid = int(task['system_pid'])
        if not psutil.pid_exists(pid):
            redis_client.client.lrem("wait_queue", 1, task_json)
            print(f"发现 GPU排队队列 中存在残余数据，已清除，进程为{pid}")

    gpu_infos = []
    gpu = {}

    devices = Device.all()  # or `Device.all()` to use NVML ordinal instead
    separator = False
    for device in devices:
        processes = device.processes()  # type: Dict[int, GpuProcess]

        gpu['index'] = device.physical_index
        gpu['GPU utilization'] = f'{device.gpu_utilization()}%'
        gpu['Total memory'] = f'{device.memory_total_human()}'
        gpu['Used memory'] = f'{device.memory_used_human()}'
        gpu['Free memory'] = f'{device.memory_free_human()}'

        keys = redis_client.client.keys()
        for key in keys:
            if 'GPU info --> ' + str(device.physical_index) in key:
                redis_client.client.delete(key)

        gpu_name = 'GPU info --> ' + str(device.physical_index) + f' utilization {device.gpu_utilization()}%  Free memory {device.memory_free_human()}'
        redis_client.client.set(gpu_name, json.dumps(gpu))

        new_processes = []
        if len(processes) > 0:
            processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
            processes.sort(key=lambda process: (process.username, process.pid))
            new_processes = []
            for snapshot in processes:
                process = {}
                process['pid'] = snapshot.pid
                process['username'] = snapshot.username
                process['time'] = snapshot.running_time_human
                process['gpu_memory'] = (snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A')
                process['gpu_memory_percent'] = f'{snapshot.gpu_memory_percent}%'
                process['command'] = snapshot.command
                new_processes.append(process)
        if len(new_processes) > 0:
            redis_client.client.set('GPU ' + str(device.physical_index) + ' processes', json.dumps(new_processes))
        else:
            redis_client.client.delete('GPU ' + str(device.physical_index) + ' processes')


    time.sleep(3)


