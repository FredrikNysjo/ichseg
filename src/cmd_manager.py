"""
.. module:: cmd_manager
   :platform: Linux, Windows
   :synopsis: Manager for handling command stack for apply/undo

.. moduleauthor:: Fredrik Nysjo
"""


class CmdManager:
    def __init__(self):
        self.stack = []
        self.max_undo_length = 8
        self.dirty = True

    def push_apply(self, cmd):
        """Apply command and push it onto the manager's undo stack"""
        if len(self.stack) >= self.max_undo_length:
            self.stack.pop(0)  # Make space on the stack
        self.stack.append(cmd.apply())
        self.dirty = True

    def pop_undo(self):
        """Pop command from the manager's undo stack and undo it"""
        if len(self.stack):
            self.stack.pop().undo()
        self.dirty = True

    def clear_stack(self):
        """Clear all commands from the manager's undo stack"""
        self.stack = []
        self.dirty = True
