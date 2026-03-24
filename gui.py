import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import asyncio

class AudiobookGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Irvine Director - Private Radio Engine v1.0")
        self.root.geometry("600x450")

        # --- Line 1: Path Selection ---
        tk.Label(root, text="PDF Path:").grid(row=0, column=0, padx=10, pady=10)
        self.path_var = tk.StringVar()
        tk.Entry(root, textvariable=self.path_var, width=50).grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_path).grid(row=0, column=2)

        # --- Line 2: Mode Selection ---
        tk.Label(root, text="Content Type:").grid(row=1, column=0, padx=10, pady=10)
        self.type_var = tk.StringVar(value="politic")
        ttk.Combobox(root, textvariable=self.type_var, values=["politic", "history", "news"]).grid(row=1, column=1, sticky="w")

        # --- Line 3: 30-minute Reinforcement Switch ---
        self.long_mode = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Forced 30-min long chapter mode (6000 words)", variable=self.long_mode).grid(row=2, column=1, sticky="w")

        # --- Line 4: Log Output ---
        self.log_text = tk.Text(root, height=15, width=70, state='disabled')
        self.log_text.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        # --- Line 5: Start Button ---
        self.start_btn = tk.Button(root, text="🚀 Start Processing (Generate Audiobook)", bg="green", fg="white", command=self.start_process)
        self.start_btn.grid(row=4, column=1, pady=10)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def browse_path(self):
        path = filedialog.askdirectory() if self.long_mode.get() else filedialog.askopenfilename()
        self.path_var.set(path)

    def start_process(self):
        # Start a thread to run asyncio to prevent the UI from freezing
        thread = threading.Thread(target=self.run_async_main)
        thread.start()

    def run_async_main(self):
        self.log("Initializing engine...")
        # Here we would call the original main() logic with GUI parameters
        # asyncio.run(your_main_logic(self.path_var.get(), self.type_var.get(), self.long_mode.get()))
        self.log("Task completed! Please check output_audiobook folder.")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AudiobookGUI(root)
    root.mainloop()