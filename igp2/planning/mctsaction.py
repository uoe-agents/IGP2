from igp2.planlibrary.macro_action import MacroAction


class MCTSAction:
    """ Wrapper class for handling actions in MCTS. """

    def __init__(self, macro_action_type: type(MacroAction), ma_args: dict):
        """ Initialise a new Action.

        Args:
            macro_action_type: The type of the macro action represented by this Action
            ma_args: The arguments that can be used to initialise a new macro action
        """
        self.macro_action_type = macro_action_type
        self.ma_args = ma_args

    def __repr__(self):
        args_str = str(self.ma_args)[1:-1].replace("\'", "").replace("array", "")
        return f"{self.macro_action_type.__name__}({args_str})"

    def __str__(self) -> str:
        return self.__repr__()
