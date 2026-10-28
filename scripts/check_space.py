import shutil
total, used, free = shutil.disk_usage("D:")
print(f"Total: {total / (1024**3):.2f} GB, Used: {used / (1024**3):.2f} GB, Free: {free / (1024**3):.2f} GB")
