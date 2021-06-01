import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox


def askopenfilename(filetypes):
    """Show an open dialog asking for a filename
    
    Returns: string with selected filename, otherwise None
    """
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    filepath = tk.filedialog.askopenfilename(filetypes=filetypes)
    return filepath


def asksaveasfilename(filetypes, defaultextension):
    """Show a save dialog asking for a filename
    
    Returns: string with selected filename, otherwise None
    """
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    filepath = tk.filedialog.asksaveasfilename(
        filetypes=filetypes, defaultextension=defaultextension
    )
    return filepath


def askokcancel(title, msg):
    """Show a confirmation dialog with options 'OK' and 'Cancel'
    
    Returns: True if 'OK' was pressed, otherwise False
    """
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    return tk.messagebox.askokcancel(title, msg)