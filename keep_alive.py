import subprocess
import time
import sys

def run_bot():
    print("🚀 Starting Research Assistant Bot Monitor...")
    while True:
        try:
            # Start the main bot process
            process = subprocess.Popen([sys.executable, "main.py"])
            print(f"✅ Bot started with PID: {process.pid}")
            
            # Wait for the process to finish (or crash)
            process.wait()
            
            if process.returncode != 0:
                print(f"⚠️ Bot crashed with exit code {process.returncode}. Restarting in 5 seconds...")
            else:
                print("ℹ️ Bot stopped normally. Restarting in 5 seconds...")
                
        except Exception as e:
            print(f"❌ Monitor error: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    run_bot()
