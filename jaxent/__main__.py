import psutil
import GPUtil



def main():
    # CPU information
    print("CPU Information:")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total cores: {psutil.cpu_count(logical=True)}")
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # GPU information
    print("\nGPU Information:")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"ID: {gpu.id}, Name: {gpu.name}")
        print(f"Load: {gpu.load*100}%")
        print(f"Free Memory: {gpu.memoryFree}MB")
        print(f"Used Memory: {gpu.memoryUsed}MB")
        print(f"Total Memory: {gpu.memoryTotal}MB")
        print(f"Temperature: {gpu.temperature} °C")



if __name__ == "__main__":
    main()

